"""Discover parts by sequential RANSAC SE(3) over a temporal window.

Uses the same sequential RANSAC as Point2Pose's registration
(`SVDClusterRANSACRegister._RANSAC`, Lin et al., ECCV 2026, arXiv:2604.10415),
but fits each consensus set between the OLDEST and NEWEST position in a sliding
window instead of between consecutive frames.

Both changes come from measurements on the earlier detectors:

* per-frame consensus sets churn -- frame-to-frame membership Jaccard was 0.34 on
  cabinet05, so no candidate ever survived long enough to spawn.  A window
  baseline accumulates the articulation instead of re-deciding every frame.
* the pairwise-rigidity detector integrates over time nicely but summarises each
  pair as a single scalar, discarding the rigid-motion model.  RANSAC SE(3) uses
  the full model -- three points determine a transform and the inlier count
  aggregates the evidence -- which is far more statistically efficient on the
  same data.  Measured coverage was 80% for the RANSAC-based consensus detector
  against 60% for pairwise rigidity.

No base is assumed: every consensus set is a rigid motion in its own right.  The
largest simply stays with the parent, which is bookkeeping, not a claim that it
is static.  Objects where every part moves (pliers, scissors) are representable.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class WindowRansacPartDiscovery:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.window = int(cfg.get("window", 45))
        self.min_history = int(cfg.get("min_history", 8))
        self.min_points = int(cfg.get("min_points", 10))
        self.update_every = int(cfg.get("update_every", 3))
        self.stable_updates = int(cfg.get("stable_updates", 3))
        self.match_overlap = float(cfg.get("match_overlap", 0.5))
        self.max_parts_total = int(cfg.get("max_parts_total", 5))
        self.stale_updates = int(cfg.get("stale_updates", 4))

        # RANSAC over the window baseline. The threshold is on the residual after
        # a rigid fit, so it only has to cover sensor noise, not the motion.
        self.inlier_thres = float(cfg.get("inlier_thres", 0.012))
        self.ransac_iters = int(cfg.get("ransac_iters", 200))
        self.sample_size = int(cfg.get("sample_size", 4))
        self.max_clusters = int(cfg.get("max_clusters", 5))
        # a candidate must move relative to the largest set to count as a part
        self.min_rel_trans = float(cfg.get("min_rel_trans", 0.012))
        self.min_rel_rot_deg = float(cfg.get("min_rel_rot_deg", 4.0))
        self.verbose = bool(cfg.get("verbose", False))

        self._hist = {}
        self._cands = {}
        self._tick = {}
        self._next_cid = 0
        self.total_spawned = 0
        self.stats_log = []
        self._reg = None

    # ---------- bookkeeping ----------

    def _register(self):
        if self._reg is None:
            from point2pose.modules.register.svd_cluster_ransac_register import (
                SVDClusterRANSACRegister,
            )
            self._reg = SVDClusterRANSACRegister({
                "ransac_iters": self.ransac_iters,
                "sample_size": self.sample_size,
                "inlier_thres": self.inlier_thres,
                "min_inliers": self.min_points,
                "max_clusters": self.max_clusters,
                "use_uncertainty": False,
            })
        return self._reg

    def observe(self, frame_id, obj_id, obj, track_table):
        idxs = track_table.obj2track_map.get(obj_id, None)
        if idxs is None or len(idxs) == 0:
            return
        idxs = np.asarray(idxs).reshape(-1)
        idxs = idxs[idxs < len(track_table.track_3d)]
        if idxs.size == 0:
            return
        p = np.asarray(track_table.track_3d[idxs], dtype=np.float64)
        ok = (np.all(np.isfinite(p), axis=1) & track_table.valid[idxs]
              & track_table.visible[idxs])
        h = self._hist.setdefault(obj_id, {})
        for t, keep, pt in zip(idxs.tolist(), ok.tolist(), p):
            if not keep:
                continue
            d = h.setdefault(int(t), [])
            d.append((int(frame_id), pt))          # keep WHEN, not just where
            if len(d) > self.window:
                d.pop(0)

    # ---------- main ----------

    @staticmethod
    def _rel_motion(Ta, Tb):
        rel = np.linalg.inv(Ta) @ Tb
        t = float(np.linalg.norm(rel[:3, 3]))
        r = float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(rel[:3, :3]).as_rotvec())))
        return t, r

    def update(self, frame_id, obj_id, obj, track_table):
        if not self.enabled or self.total_spawned >= self.max_parts_total:
            return []
        n = self._tick.get(obj_id, 0) + 1
        self._tick[obj_id] = n
        if n % self.update_every != 0:
            return []

        h = self._hist.get(obj_id, {})
        if not h:
            return []

        # A single SE(3) is only meaningful between two FIXED instants. Letting
        # every track use its own first/last sample compared different time
        # intervals against each other, which is simply the wrong fit.
        newest = max(d[-1][0] for d in h.values() if d)

        # A FIXED-LENGTH trailing window, in frame units. Deriving t0 from the
        # tracks present (a median of their first frames) made the interval move
        # as new keypoints appeared, so consecutive updates compared different
        # motions and the consensus sets legitimately disagreed: on noise-free
        # data the largest cluster matched the previous one at Jaccard 0.35 and
        # swung 124 -> 220 -> 80 tracks. The frame stride is unknown here, so it
        # is estimated from the history itself.
        steps = []
        for d in h.values():
            if len(d) >= 2:
                steps.append(d[-1][0] - d[-2][0])
        stride = int(np.median(steps)) if steps else 1
        stride = max(1, stride)

        def gather(span):
            t0 = newest - span
            tol = max(stride, span // 6)
            tt, aa, bb = [], [], []
            for t, d in h.items():
                if len(d) < self.min_history:
                    continue
                a = b = None
                da = db = None
                for f, pt in d:
                    if da is None or abs(f - t0) < da:
                        a, da = pt, abs(f - t0)
                    if db is None or abs(f - newest) < db:
                        b, db = pt, abs(f - newest)
                if a is None or b is None or da > tol or db > tol:
                    continue
                tt.append(t); aa.append(a); bb.append(b)
            return tt, aa, bb, t0

        # shrink the window until enough tracks span it, so early frames and
        # freshly seeded parts still get measured instead of being skipped
        span = self.window * stride
        tids = []
        while span >= 4 * stride:
            tids, src, dst, t0 = gather(span)
            if len(tids) >= 2 * self.min_points:
                break
            span //= 2
        if len(tids) < 2 * self.min_points:
            self.stats_log.append({
                "frame": int(frame_id), "obj": int(obj_id), "n_clusters": 0,
                "sizes": [], "n_tracks": len(tids), "n_history": len(h),
                "skipped": "too few tracks spanning a common window",
            })
            return []
        src = np.stack(src); dst = np.stack(dst)
        tids = np.asarray(tids, dtype=np.int64)

        reg = self._register()
        remaining = np.ones(len(src), dtype=bool)
        clusters = []
        for _ in range(self.max_clusters):
            c = reg._RANSAC(p0=src, tgt_pcd=dst, w=None,
                            remaining=remaining, init_pose=None)
            if c is None:
                break
            clusters.append(c)
        clusters.sort(key=lambda c: -c["ninliers"])

        rec = {"frame": int(frame_id), "obj": int(obj_id),
               "n_tracks": int(len(tids)),
               "baseline_frames": int(newest - t0),
               "n_clusters": len(clusters),
               "sizes": [int(c["ninliers"]) for c in clusters]}
        self.stats_log.append(rec)
        if len(clusters) < 2:
            return []

        T_main = clusters[0]["T"]
        rec["cluster_members"] = [
            [int(x) for x in tids[c["inliers"]].tolist()] for c in clusters]

        cands = self._cands.setdefault(obj_id, {})
        spawn, matched = [], set()
        for c in clusters[1:]:
            members = set(int(x) for x in tids[c["inliers"]].tolist())
            if len(members) < self.min_points:
                continue
            tr, rot = self._rel_motion(T_main, c["T"])
            moving = tr >= self.min_rel_trans or rot >= self.min_rel_rot_deg

            best_cid, best_j = None, 0.0
            for cid, cand in cands.items():
                if cid in matched:
                    continue
                j = len(members & cand["members"]) / len(members | cand["members"])
                if j > best_j:
                    best_cid, best_j = cid, j

            if best_cid is not None and best_j >= self.match_overlap:
                matched.add(best_cid)
                cand = cands[best_cid]
                cand["members"] = members
                cand["age"] += 1 if moving else 0
                cand["last"] = n
                cid = best_cid
            else:
                cid = self._next_cid
                self._next_cid += 1
                cands[cid] = {"members": members, "age": 1 if moving else 0,
                              "last": n}
                matched.add(cid)

            if cands[cid]["age"] >= self.stable_updates and \
                    self.total_spawned < self.max_parts_total:
                spawn.append(np.array(sorted(cands[cid]["members"]), dtype=np.int64))
                self.total_spawned += 1
                cands.pop(cid, None)
                matched.discard(cid)
                if self.verbose:
                    print(f"[WindowRansac] frame {frame_id} obj {obj_id}: "
                          f"{len(members)} tracks promoted "
                          f"(rel {tr*1000:.0f}mm / {rot:.1f}deg over "
                          f"{rec['baseline_frames']} frames)")

        for cid in [c for c, v in cands.items() if n - v["last"] > self.stale_updates]:
            cands.pop(cid, None)
        rec["n_candidates"] = len(cands)
        rec["max_age"] = int(max((v["age"] for v in cands.values()), default=0))
        return spawn

    def note_new_object(self, obj_id):
        self._hist.setdefault(obj_id, {})
        self._cands.setdefault(obj_id, {})

    def drop_tracks(self, obj_id, track_ids):
        h = self._hist.get(obj_id, None)
        if not h:
            return
        for t in np.asarray(track_ids).reshape(-1).tolist():
            h.pop(int(t), None)

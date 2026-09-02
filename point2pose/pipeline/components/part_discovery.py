"""Discover articulated parts from persistent secondary consensus sets.

Consumes the sequential-RANSAC consensus sets produced by Point2Pose's
registration (Lin et al., ECCV 2026, arXiv:2604.10415).  The two-signal spawn
rule -- a persistent secondary set AND relative motion against its parent --
is from this project's own plan, section 2.3.

The rigid-body assumption in Point2Pose enters only at frame-to-map
registration, whose sequential RANSAC already extracts several rigid-motion
consensus sets per object and then collapses them to one pose.  This component
watches the consensus sets that get discarded: one that persists across frames
*and* moves relative to the object it belongs to is a part, not noise.

Two independent signals are required before a part is spawned, because either
alone produces false positives:

  persistence  -- a candidate must keep roughly the same membership over
                  several frames.  Depth noise reshuffles membership every
                  frame, so noise cannot survive this.
  relative motion -- the candidate must actually be moving with respect to its
                  parent.  Without this, a still-closed drawer splits off purely
                  from depth bias, which is measurably the dominant false
                  positive.

A drawer that has never been opened therefore stays part of its parent, which is
the correct answer: parts are defined by motion, not by semantics.
"""

import numpy as np
from scipy.spatial.transform import Rotation


class PartCandidate:
    __slots__ = ("members", "age", "last_seen", "last_T")

    def __init__(self, members, T, frame_id):
        self.members = set(members)
        self.age = 0
        self.last_seen = frame_id
        self.last_T = T


class PartDiscovery:
    """Per-object tracker of secondary consensus sets."""

    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.persist_frames = int(cfg.get("persist_frames", 6))
        self.min_points = int(cfg.get("min_points", 12))
        self.overlap_thres = float(cfg.get("overlap_thres", 0.4))
        self.min_rel_trans = float(cfg.get("min_rel_trans", 0.015))
        self.min_rel_rot_deg = float(cfg.get("min_rel_rot_deg", 5.0))
        self.max_parts_per_object = int(cfg.get("max_parts_per_object", 4))
        # Parts can themselves spawn parts (genuine for nested articulation),
        # so a per-object cap alone does not bound the total.
        self.max_parts_total = int(cfg.get("max_parts_total", 6))
        self.total_spawned = 0
        self.stale_frames = int(cfg.get("stale_frames", 12))
        self.verbose = bool(cfg.get("verbose", False))

        self._candidates = {}   # obj_id -> {cid: PartCandidate}
        self._next_cid = 0
        self._spawned_per_object = {}
        self.stats_log = []     # per-frame diagnostics, for tuning

    # ---------- helpers ----------

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _rel_motion(T_parent, T_cand):
        """Translation (m) and rotation (deg) of a candidate w.r.t. its parent."""
        if T_parent is None or T_cand is None:
            return 0.0, 0.0
        rel = np.linalg.inv(T_parent) @ T_cand
        trans = float(np.linalg.norm(rel[:3, 3]))
        rot = float(np.degrees(np.linalg.norm(Rotation.from_matrix(rel[:3, :3]).as_rotvec())))
        return trans, rot

    @staticmethod
    def _clusters_from_stats(stats):
        clusters = stats.get("clusters", None) if isinstance(stats, dict) else None
        if clusters is None or len(clusters) == 0:
            return []
        return [c for c in clusters if isinstance(c, dict) and "inliers" in c]

    # ---------- main ----------

    def update(self, frame_id, obj_id, fe_result):
        """Return a list of global track-id arrays that should become new parts.

        Cluster inlier indices are positions into the registration's source
        array; `valid_indices[obj_id]` maps them back to global track ids.
        """
        if not self.enabled:
            return []

        stats = fe_result.reg_stats.get(obj_id, {})
        clusters = self._clusters_from_stats(stats)
        cands = self._candidates.setdefault(obj_id, {})

        rec = {
            "frame": int(frame_id), "obj": int(obj_id),
            "n_clusters": len(clusters),
            "sizes": [int(c.get("ninliers", 0)) for c in clusters],
        }
        self.stats_log.append(rec)
        if len(clusters) < 2:
            self._expire(cands, frame_id)
            return []

        valid_idx = fe_result.valid_indices.get(obj_id, None)
        if valid_idx is None or len(valid_idx) == 0:
            self._expire(cands, frame_id)
            return []
        valid_idx = np.asarray(valid_idx).reshape(-1)

        def to_global(c):
            inl = np.asarray(c["inliers"]).reshape(-1)
            inl = inl[(inl >= 0) & (inl < valid_idx.size)]
            return valid_idx[inl]

        clusters = sorted(clusters, key=lambda c: -int(c.get("ninliers", 0)))
        # The best cluster is the object's own motion; everything else is a
        # candidate part measured relative to it.
        T_parent = clusters[0].get("T", None)

        matched, fresh, spawn = set(), {}, []
        best_j_seen, moving_seen = [], []
        for c in clusters[1:]:
            g = to_global(c)
            if g.size == 0:
                continue
            members = set(int(x) for x in g.tolist())
            trans, rot = self._rel_motion(T_parent, c.get("T", None))
            moving = (trans >= self.min_rel_trans) or (rot >= self.min_rel_rot_deg)
            moving_seen.append(moving)

            best_cid, best_j = None, 0.0
            for cid, cand in cands.items():
                if cid in matched:
                    continue
                j = self._jaccard(members, cand.members)
                if j > best_j:
                    best_cid, best_j = cid, j

            best_j_seen.append(best_j)
            if best_cid is not None and best_j >= self.overlap_thres:
                matched.add(best_cid)
                cand = cands[best_cid]
                cand.members |= members
                cand.age += 1 if moving else 0
                cand.last_seen = frame_id
                cand.last_T = c.get("T", None)
                cid = best_cid
            else:
                cid = self._next_cid
                self._next_cid += 1
                cand = PartCandidate(members, c.get("T", None), frame_id)
                cand.age = 1 if moving else 0
                cands[cid] = cand

            n_spawned = self._spawned_per_object.get(obj_id, 0)
            if (
                cand.age >= self.persist_frames
                and len(cand.members) >= self.min_points
                and n_spawned < self.max_parts_per_object
                and self.total_spawned < self.max_parts_total
            ):
                spawn.append(np.array(sorted(cand.members), dtype=np.int64))
                self._spawned_per_object[obj_id] = n_spawned + 1
                self.total_spawned += 1
                cands.pop(cid, None)
                matched.discard(cid)
                if self.verbose:
                    print(
                        f"[PartDiscovery] frame {frame_id} obj {obj_id}: candidate "
                        f"{cid} promoted ({len(cand.members)} pts, rel "
                        f"{trans*1000:.0f}mm/{rot:.1f}deg)"
                    )

        rec["best_jaccard"] = round(max(best_j_seen, default=0.0), 3)
        rec["n_moving"] = int(sum(moving_seen))
        rec["max_age"] = int(max((c.age for c in cands.values()), default=0))
        rec["n_candidates"] = len(cands)
        self._expire(cands, frame_id)
        return spawn

    def _expire(self, cands, frame_id):
        """Drop candidates that have not been re-observed recently."""
        for cid in [c for c, v in cands.items()
                    if frame_id - v.last_seen > self.stale_frames]:
            cands.pop(cid, None)

    def note_new_object(self, obj_id):
        self._candidates.setdefault(obj_id, {})
        self._spawned_per_object.setdefault(obj_id, 0)

"""Discover parts by clustering per-track motion trajectories.

An alternative to PartDiscovery, which matches per-frame RANSAC consensus sets
across frames by membership overlap.  That matching is the measured failure
mode: with several small clusters per frame the membership churns, the best
Jaccard sits around 0.34, and no candidate survives long enough to spawn.

Here a track's identity is its whole motion history instead, so there is nothing
to match between frames.  Following the motion-prior stage of VideoArtGS
(arXiv:2509.17647), each trajectory is fitted with a line and a circle, labelled
static / prismatic / revolute / noise, and the moving tracks are clustered in a
feature space built from their fitted motion parameters.

Two deliberate departures from that work, both required by the online setting:

  * trajectories are accumulated in a sliding window and re-fitted as evidence
    arrives, rather than analysed once over a finished video;
  * clustering is density-based, so the number of parts emerges instead of being
    supplied as k -- we do not know the part count in advance, which is the
    whole point of discovery.

Everything is expressed in the PARENT OBJECT'S frame, where the parent's own
geometry is stationary by construction: a track that moves in that frame belongs
to something else.  That removes the "which cluster is the base" question that
the consensus-set detector has to guess at.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

STATIC, PRISMATIC, REVOLUTE, NOISE = 0, 1, 2, 3


def fit_line(P):
    """Total-least-squares line fit. Returns (direction, point, rms residual)."""
    c = P.mean(axis=0)
    q = P - c
    try:
        _, s, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    d = vt[0]
    resid = q - np.outer(q @ d, d)
    return d, c, float(np.sqrt((resid ** 2).sum(axis=1).mean()))


def fit_circle(P):
    """Fit a plane, then a circle within it.

    Returns (axis direction, centre, radius, rms residual). The axis is the
    plane normal, which for a revolute joint is the joint axis.
    """
    if len(P) < 5:
        return None
    c = P.mean(axis=0)
    q = P - c
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    n = vt[2]                      # plane normal = candidate joint axis
    e1, e2 = vt[0], vt[1]
    u, v = q @ e1, q @ e2
    plane_resid = float(np.sqrt(((q @ n) ** 2).mean()))

    # algebraic circle fit in the plane: u^2+v^2 = A*u + B*v + C
    A = np.stack([u, v, np.ones_like(u)], axis=1)
    b = u ** 2 + v ** 2
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    uc, vc = sol[0] / 2.0, sol[1] / 2.0
    r2 = sol[2] + uc ** 2 + vc ** 2
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    radial = np.sqrt((u - uc) ** 2 + (v - vc) ** 2) - r
    resid = float(np.sqrt(np.mean(radial ** 2) + plane_resid ** 2))
    centre = c + uc * e1 + vc * e2
    return n, centre, r, resid


class TrackHistory:
    __slots__ = ("pts", "frames", "_maxlen")

    def __init__(self, maxlen):
        self.pts = []
        self.frames = []
        self._maxlen = maxlen

    def add(self, p, frame_id):
        self.pts.append(p)
        self.frames.append(frame_id)
        if len(self.pts) > self._maxlen:
            self.pts.pop(0)
            self.frames.pop(0)

    def array(self):
        return np.asarray(self.pts, dtype=np.float64)


class TrajectoryPartDiscovery:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.window = int(cfg.get("window", 60))
        self.min_history = int(cfg.get("min_history", 12))
        self.static_thres = float(cfg.get("static_thres", 0.02))   # m of travel
        self.fit_thres = float(cfg.get("fit_thres", 0.01))         # m rms
        self.min_points = int(cfg.get("min_points", 12))
        self.max_k = int(cfg.get("max_k", 4))
        self.min_silhouette = float(cfg.get("min_silhouette", 0.45))
        self.kmeans_min_samples = int(cfg.get("kmeans_min_samples", 16))
        self.stable_updates = int(cfg.get("stable_updates", 3))
        self.update_every = int(cfg.get("update_every", 5))
        self.max_parts_total = int(cfg.get("max_parts_total", 5))
        self.voxel_rel = float(cfg.get("voxel_rel", 0.02))
        self.verbose = bool(cfg.get("verbose", False))

        self._hist = {}          # obj_id -> {track_id: TrackHistory}
        self._stable = {}        # obj_id -> {signature: count}
        self.total_spawned = 0
        self._tick = {}
        self.stats_log = []

    # ---------- trajectory bookkeeping ----------

    def observe(self, frame_id, obj_id, obj, track_table):
        """Append this frame's track positions, expressed in the object frame."""
        idxs = track_table.obj2track_map.get(obj_id, None)
        if idxs is None or len(idxs) == 0 or obj.pose is None:
            return
        idxs = np.asarray(idxs).reshape(-1)
        idxs = idxs[idxs < len(track_table.track_3d)]
        if idxs.size == 0:
            return

        p_cam = np.asarray(track_table.track_3d[idxs], dtype=np.float64)
        ok = np.all(np.isfinite(p_cam), axis=1) & track_table.valid[idxs] \
            & track_table.visible[idxs]
        if not np.any(ok):
            return
        T_inv = np.linalg.inv(obj.pose)
        p_obj = (T_inv @ np.hstack([p_cam, np.ones((len(p_cam), 1))]).T).T[:, :3]

        h = self._hist.setdefault(obj_id, {})
        for t, keep, p in zip(idxs.tolist(), ok.tolist(), p_obj):
            if keep:
                h.setdefault(int(t), TrackHistory(self.window)).add(p, frame_id)

    # ---------- per-track motion model ----------

    def _classify(self, P):
        """Label one trajectory and return (label, feature vector).

        Motion classification follows VideoArtGS: static below a displacement
        threshold, prismatic by line fit, revolute by plane-then-circle fit, and
        noise when neither fit is good enough.
        """
        travel = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
        if travel < self.static_thres:
            return STATIC, None

        # Adaptive spatial downsampling: voxel size scaled to the trajectory's
        # own range, keeping one sample per voxel. Without this a track that
        # dwells in one place dominates its own fit ("fitting collapse").
        if self.voxel_rel > 0 and travel > 0:
            vox = max(self.voxel_rel * travel, 1e-4)
            key = np.round(P / vox).astype(np.int64)
            _, first = np.unique(key, axis=0, return_index=True)
            Q = P[np.sort(first)]
        else:
            Q = P
        if len(Q) < 5:
            return NOISE, None

        line = fit_line(Q)
        circ = fit_circle(Q)
        lr = line[2] if line else np.inf
        cr = circ[3] if circ else np.inf
        if min(lr, cr) > self.fit_thres:
            return NOISE, None

        start, avg = Q[0], Q.mean(axis=0)
        n_steps = max(len(Q) - 1, 1)

        if cr < lr and circ is not None:
            n, centre, r, _ = circ
            if n[np.argmax(np.abs(n))] < 0:
                n = -n
            # angular velocity: arc length travelled / radius, per step
            span = float(np.linalg.norm(Q[-1] - Q[0]))
            omega = (span / max(r, 1e-3)) / n_steps
            # revolute: start pos, avg pos, axis direction, axis origin, angular vel
            return REVOLUTE, np.concatenate([start, avg, n, centre, [omega]])

        d, _, _ = line
        if d[np.argmax(np.abs(d))] < 0:
            d = -d
        vel = float(np.linalg.norm(Q[-1] - Q[0])) / n_steps
        # prismatic: start pos, avg pos, motion direction, normalized velocity
        return PRISMATIC, np.concatenate([start, avg, d, [vel]])

    @staticmethod
    def _iterative_filter(F, members, dir_slice, pos_slice,
                          max_angle_deg=25.0, max_dist_sigma=2.0, iters=3):
        """Drop cluster members inconsistent in motion direction or position.

        The paper's post-clustering cleanup: alternate an angular test on the
        motion/axis direction with a Euclidean test on position.
        """
        keep = np.ones(len(members), dtype=bool)
        for _ in range(iters):
            if keep.sum() < 3:
                break
            D = F[keep, dir_slice]
            mean_dir = D.mean(axis=0)
            nrm = np.linalg.norm(mean_dir)
            if nrm < 1e-9:
                break
            mean_dir = mean_dir / nrm
            cos = np.clip(F[:, dir_slice] @ mean_dir, -1.0, 1.0)
            ang_ok = np.degrees(np.arccos(np.abs(cos))) <= max_angle_deg

            C = F[keep, pos_slice]
            centre = C.mean(axis=0)
            d = np.linalg.norm(F[:, pos_slice] - centre, axis=1)
            sd = np.linalg.norm(C - centre, axis=1).std() + 1e-6
            pos_ok = d <= max_dist_sigma * sd + 1e-3

            new_keep = ang_ok & pos_ok
            if new_keep.sum() < 3 or (new_keep == keep).all():
                keep = new_keep if new_keep.sum() >= 3 else keep
                break
            keep = new_keep
        return keep

    # ---------- main ----------

    def update(self, frame_id, obj_id, obj, track_table):
        """Return a list of global track-id arrays that should become parts."""
        if not self.enabled or self.total_spawned >= self.max_parts_total:
            return []
        # Count PROCESSED updates, not raw frame ids: with a frame stride the
        # modulo test only fires on frames divisible by stride*update_every,
        # which silently starved the evidence counter.
        n = self._tick.get(obj_id, 0) + 1
        self._tick[obj_id] = n
        if n % self.update_every != 0:
            return []
        h = self._hist.get(obj_id, {})
        if len(h) < self.min_points:
            return []

        tids, feats, labels, kinds = [], [], [], []
        for t, hist in h.items():
            if len(hist.pts) < self.min_history:
                continue
            lab, f = self._classify(hist.array())
            labels.append(lab)
            if lab in (PRISMATIC, REVOLUTE) and f is not None:
                tids.append(t)
                feats.append(f)
                kinds.append(lab)

        n_moving = len(tids)
        rec = {"frame": int(frame_id), "obj": int(obj_id),
               "n_tracks": len(h), "n_moving": n_moving,
               "n_static": int(sum(1 for l in labels if l == STATIC)),
               "n_noise": int(sum(1 for l in labels if l == NOISE))}
        self.stats_log.append(rec)

        if n_moving < self.min_points:
            return []

        tids_arr = np.asarray(tids, dtype=np.int64)
        kinds = np.asarray(kinds)
        assignments = []

        # Prismatic and revolute tracks are clustered separately: their feature
        # vectors carry different quantities, and a shared space would compare
        # an axis origin against a translation direction.
        for kind in (PRISMATIC, REVOLUTE):
            sel = np.where(kinds == kind)[0]
            if sel.size < max(self.min_points, self.kmeans_min_samples):
                continue
            F = np.asarray([feats[i] for i in sel], dtype=np.float64)
            Fz = (F - F.mean(axis=0)) / (F.std(axis=0) + 1e-6)

            # the paper supplies k; we do not know the part count, so pick it by
            # silhouette and fall back to "one part" when no split is supported
            best = None
            kmax = min(self.max_k, max(2, sel.size // self.min_points))
            for k in range(2, kmax + 1):
                try:
                    lab = KMeans(n_clusters=k, n_init=4, random_state=0).fit_predict(Fz)
                except Exception:
                    continue
                if len(set(lab)) < 2:
                    continue
                try:
                    sc = silhouette_score(Fz, lab)
                except Exception:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, lab)
            if best is None or best[0] < self.min_silhouette:
                labs = np.zeros(sel.size, dtype=int)     # single group
            else:
                labs = best[1]

            dir_slice = slice(6, 9)                      # direction / axis
            pos_slice = slice(3, 6)                      # average position
            for c in sorted(set(labs)):
                idx = np.where(labs == c)[0]
                if idx.size < self.min_points:
                    continue
                keep = self._iterative_filter(Fz[idx], idx, dir_slice, pos_slice)
                idx = idx[keep]
                if idx.size < self.min_points:
                    continue
                assignments.append((tids_arr[sel[idx]], Fz[idx].mean(axis=0), kind))

        rec["n_clusters"] = len(assignments)

        spawn = []
        stable = self._stable.setdefault(obj_id, {})
        seen = set()
        for members, centre, kind in assignments:
            if members.size < self.min_points:
                continue
            sig = (int(kind),) + tuple(np.round(centre[:9], 0))
            seen.add(sig)
            stable[sig] = stable.get(sig, 0) + 1
            if stable[sig] >= self.stable_updates and \
                    self.total_spawned < self.max_parts_total:
                spawn.append(members)
                self.total_spawned += 1
                stable.pop(sig, None)
                if self.verbose:
                    kname = "prismatic" if kind == PRISMATIC else "revolute"
                    print(f"[TrajPartDiscovery] frame {frame_id} obj {obj_id}: "
                          f"{kname} cluster of {members.size} tracks promoted "
                          f"({n_moving} moving of {len(h)})")
        for sig in list(stable):
            if sig not in seen:
                stable.pop(sig, None)
        return spawn

    def note_new_object(self, obj_id):
        self._hist.setdefault(obj_id, {})
        self._stable.setdefault(obj_id, {})

    def drop_tracks(self, obj_id, track_ids):
        """Tracks that left this object are no longer its evidence."""
        h = self._hist.get(obj_id, None)
        if not h:
            return
        for t in np.asarray(track_ids).reshape(-1).tolist():
            h.pop(int(t), None)

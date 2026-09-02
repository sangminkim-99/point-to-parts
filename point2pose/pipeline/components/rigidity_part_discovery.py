"""Discover parts by clustering tracks on pairwise rigidity.

Method, not novel: build an affinity matrix from pairwise distances between
point trajectories and run spectral clustering on it.  That is the construction
of Brox and Malik, "Object Segmentation by Long Term Analysis of Point
Trajectories", ECCV 2010, pp. 282-295.  The related observation that the
trajectories of one rigid motion span a low-dimensional affine subspace comes
from the multibody-factorization / subspace-clustering line (Costeira & Kanade;
sparse subspace clustering).  A causal variant of motion segmentation also
already exists (arXiv:1604.00136).  See the component-provenance note.

Why this affinity rather than the motion-feature clustering of
trajectory_part_discovery.py: two points on the same rigid body keep a constant
distance no matter what that body does, so

    w_ij = exp( -var_t || p_i(t) - p_j(t) || / sigma^2 )

is **invariant to any motion common to both points** -- the object being carried,
the camera moving, or both.  The measured failure of the feature-space version
was exactly that the common motion dominates the features and most of the object
collapses into one cluster.  It also needs no reference frame, so there is no
"which cluster is the base" question, and no base is assumed to exist: pliers and
scissors, where both parts move, are representable.

Everything here is computed over a sliding window so it stays causal.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


class RigidityPartDiscovery:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.window = int(cfg.get("window", 45))
        self.min_history = int(cfg.get("min_history", 10))
        self.min_points = int(cfg.get("min_points", 10))
        self.update_every = int(cfg.get("update_every", 5))
        self.stable_updates = int(cfg.get("stable_updates", 3))
        self.max_parts_total = int(cfg.get("max_parts_total", 5))
        self.max_k = int(cfg.get("max_k", 4))
        # a pair is "rigid" when its distance varies by less than this (metres)
        self.sigma = float(cfg.get("sigma", 0.01))
        self.rigid_tol = float(cfg.get("rigid_tol", 0.012))
        # a split is only accepted if the groups really are non-rigid to each other
        self.min_cross_std = float(cfg.get("min_cross_std", 0.02))
        # separation must also beat the within-group noise by this factor
        self.min_ratio = float(cfg.get("min_ratio", 4.0))
        # a pair must break rigidity in this many separate windows before the
        # evidence counts: a running max alone is set by the worst noise spike,
        # which on pliers05 drove frac_rigid to 26% and shattered the object
        # into groups of 2-8 tracks
        self.min_violations = int(cfg.get("min_violations", 3))
        self.max_tracks = int(cfg.get("max_tracks", 400))
        # candidates are matched across updates by membership OVERLAP, not by an
        # exact signature: a single track joining or leaving must not reset the
        # evidence counter (that mistake is what stalled the earlier detector).
        self.match_overlap = float(cfg.get("match_overlap", 0.6))
        self.verbose = bool(cfg.get("verbose", False))

        self._hist = {}
        self._stable = {}
        # cumulative pairwise statistics, per object: Welford accumulators over
        # ALL observations rather than a sliding window.  Rigidity violation is
        # monotone evidence -- two points shown to move relative to each other
        # can never be one rigid body again -- so it must not decay when the
        # part stops moving, which is what the windowed statistic did.
        self._cum = {}
        self._cnt = {}
        self._tid_index = {}
        self.total_spawned = 0
        self._next_cid = 0
        self._tick = {}
        self.stats_log = []

    # ---------- bookkeeping ----------

    def observe(self, frame_id, obj_id, obj, track_table):
        """Record this frame's 3D positions, in the CAMERA frame.

        No object frame is used: the affinity is invariant to common motion, so
        transforming into an estimated parent pose would only inject that pose's
        error into the measurement.
        """
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
            d.append(pt)
            if len(d) > self.window:
                d.pop(0)

    # ---------- affinity ----------

    def _pairwise(self, P, V):
        """Distance-std for every pair, over that PAIR's own co-visible frames.

        Truncating all tracks to the shortest history throws the evidence away:
        the pipeline keeps adding keypoints, so one freshly created track pinned
        the window to its own length for everyone.  Measured on cabinet03, that
        used 13 of 42 available frames, and since this statistic grows with the
        observed motion the signal collapsed with it.

        P is (T, N, 3) with arbitrary values where unobserved, V is (T, N) bool.
        Returns (std, n_overlap).
        """
        T, N, _ = P.shape
        M = (V[:, :, None] & V[:, None, :]).astype(np.float64)      # (T, N, N)
        Pz = np.where(V[:, :, None], P, 0.0)
        d = np.linalg.norm(Pz[:, :, None, :] - Pz[:, None, :, :], axis=-1) * M
        n = M.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = d.sum(axis=0) / n
            var = (d ** 2).sum(axis=0) / n - mean ** 2
        std = np.sqrt(np.clip(var, 0.0, None))
        std[~np.isfinite(std)] = np.inf
        return std, n

    def _accumulate(self, obj_id, tids, S_win, nov):
        """Fold this window's pairwise stds into a per-object running maximum.

        The running max is the evidence: once a pair has been seen to break
        rigidity above the noise floor, that fact stands for the rest of the
        sequence even after the part comes to rest.
        """
        idx = self._tid_index.setdefault(obj_id, {})
        for t in tids:
            if t not in idx:
                idx[t] = len(idx)
        n = len(idx)

        C = self._cum.get(obj_id)
        if C is None or C.shape[0] < n:
            new = np.zeros((n, n), dtype=np.float64)
            if C is not None:
                new[: C.shape[0], : C.shape[1]] = C
            C = new
            self._cum[obj_id] = C

        K = self._cnt.get(obj_id)
        if K is None or K.shape[0] < n:
            newk = np.zeros((n, n), dtype=np.int32)
            if K is not None:
                newk[: K.shape[0], : K.shape[1]] = K
            K = newk
            self._cnt[obj_id] = K

        cols = np.array([idx[t] for t in tids], dtype=int)
        ok = np.isfinite(S_win) & (nov >= self.min_history)
        sub = C[np.ix_(cols, cols)]
        C[np.ix_(cols, cols)] = np.where(ok, np.maximum(sub, S_win), sub)
        subk = K[np.ix_(cols, cols)]
        K[np.ix_(cols, cols)] = subk + (ok & (S_win > self.rigid_tol)).astype(np.int32)

        acc = C[np.ix_(cols, cols)].copy()
        votes = K[np.ix_(cols, cols)]
        # pairs without repeated evidence are treated as rigid, so one noisy
        # window cannot permanently separate two points on the same body
        return np.where(votes >= self.min_violations, acc,
                        np.minimum(acc, self.rigid_tol * 0.5))

    def update(self, frame_id, obj_id, obj, track_table):
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
        if len(h) < 2 * self.min_points:
            return []

        # tracks with a full, co-visible window
        tids = [t for t, d in h.items() if len(d) >= self.min_history]
        if len(tids) < 2 * self.min_points:
            return []
        if len(tids) > self.max_tracks:                     # keep the cost bounded
            tids = list(np.random.default_rng(0).choice(
                tids, self.max_tracks, replace=False))
        # right-align every track in a common time grid; no global truncation
        L = min(max(len(h[t]) for t in tids), self.window)
        N = len(tids)
        P = np.zeros((L, N, 3), dtype=np.float64)
        V = np.zeros((L, N), dtype=bool)
        for j, t in enumerate(tids):
            a = np.asarray(h[t][-L:], dtype=np.float64)
            P[L - len(a):, j] = a
            V[L - len(a):, j] = True

        S_win, nov = self._pairwise(P, V)
        S = self._accumulate(obj_id, tids, S_win, nov)
        hist_lens = [len(h[t]) for t in tids]
        rec = {"frame": int(frame_id), "obj": int(obj_id),
               "n_tracks": len(tids),
               "L_used": int(L), "pair_overlap_median": float(np.median(nov)),
               "hist_min": int(np.min(hist_lens)),
               "hist_median": int(np.median(hist_lens)),
               "hist_max": int(np.max(hist_lens)),
               "median_pair_std": float(np.median(S)),
               "frac_rigid": float((S <= self.rigid_tol).mean())}
        self.stats_log.append(rec)

        # If almost every pair is rigid, the object is still one body. Refusing to
        # split here is what keeps a closed drawer attached to its cabinet.
        if rec["frac_rigid"] > 0.97:
            rec["n_clusters"] = 1
            return []

        # Complete-linkage agglomeration on the distance-std matrix, cut at
        # rigid_tol.  Two tracks end up together only if EVERY pair in the merged
        # group stays rigid, which is the definition of a rigid body.
        #
        # Not spectral clustering: normalized cuts prefer balanced partitions, so
        # a 20-vs-230 split -- one small drawer against the body, i.e. our actual
        # case -- is exactly what it resists.  Agglomeration is size-agnostic and
        # needs no k, which matters because the part count is what we are trying
        # to discover.
        # Average linkage, not complete: complete linkage cuts on the MAX pairwise
        # distance, so one bad track shatters a group that is otherwise rigid --
        # measured on cabinet03, where groups were split despite a median
        # between-group std of only 1.4-2.1 mm.  Average linkage is still free of
        # the chaining that single linkage suffers.
        lab = AgglomerativeClustering(
            n_clusters=None, distance_threshold=self.rigid_tol,
            metric="precomputed", linkage="average").fit_predict(S)

        sizes = np.bincount(lab)
        keep = np.where(sizes >= self.min_points)[0]
        rec["n_raw_groups"] = int(len(sizes))
        rec["group_sizes"] = [int(x) for x in np.sort(sizes)[::-1][:8]]
        if len(keep) < 2:
            rec["n_clusters"] = 1
            return []

        # tracks in groups too small to trust stay with the parent
        big = np.isin(lab, keep)
        within, cross = [], []
        for a in keep:
            ia = lab == a
            within.append(np.median(S[np.ix_(ia, ia)]))
            for b in keep:
                if b <= a:
                    continue
                cross.append(np.median(S[np.ix_(ia, lab == b)]))
        within_m, cross_m = float(np.max(within)), float(np.min(cross))
        # Judge separation against the measured noise floor rather than an
        # absolute distance: the same articulation looks very different at 0.9 m
        # on an Xtion than it would on a cleaner sensor, and an absolute gate
        # tuned on one sequence does not transfer.  Both tests must pass.
        rec["within_std"] = within_m
        rec["cross_std"] = cross_m
        rec["ratio"] = cross_m / max(within_m, 1e-6)
        if cross_m < self.min_cross_std or rec["ratio"] < self.min_ratio:
            rec["n_clusters"] = 1
            rec["rejected_cross_std"] = cross_m
            return []

        k = len(keep)
        score = cross_m - within_m
        rec.update({"n_clusters": int(k), "within_std": within_m,
                    "cross_std": cross_m, "separation": score})
        rec["cluster_members"] = [
            [int(x) for x in np.asarray(tids)[lab == c].tolist()] for c in keep
        ]
        # relabel so the loop below sees only the kept groups
        lab = np.where(big, lab, -1)

        tids = np.asarray(tids, dtype=np.int64)
        cands = self._stable.setdefault(obj_id, {})
        spawn, matched = [], set()
        # the largest group is left with the parent; the others become parts.
        # This is a bookkeeping choice only -- no claim that it is "the base".
        counts = {int(c): int((lab == c).sum()) for c in keep}
        order = sorted(counts, key=lambda c: -counts[c])
        for c in order[1:]:
            members = set(int(x) for x in tids[lab == c].tolist())
            if len(members) < self.min_points:
                continue

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
                cand["members"] = members          # follow the current grouping
                cand["age"] += 1
                cand["last_seen"] = frame_id
                cid = best_cid
            else:
                cid = self._next_cid
                self._next_cid += 1
                cands[cid] = {"members": members, "age": 1, "last_seen": frame_id}
                matched.add(cid)

            if cands[cid]["age"] >= self.stable_updates and \
                    self.total_spawned < self.max_parts_total:
                spawn.append(np.array(sorted(cands[cid]["members"]), dtype=np.int64))
                self.total_spawned += 1
                cands.pop(cid, None)
                matched.discard(cid)
                if self.verbose:
                    print(f"[RigidityPartDiscovery] frame {frame_id} obj {obj_id}: "
                          f"group of {len(members)} tracks promoted after "
                          f"{self.stable_updates} updates "
                          f"(within {within_m*1000:.1f}mm, cross {cross_m*1000:.1f}mm)")

        # forget candidates that stopped being re-observed
        for cid in [c for c, v in cands.items()
                    if frame_id - v["last_seen"] > self.update_every * 4]:
            cands.pop(cid, None)
        rec["n_candidates"] = len(cands)
        rec["max_age"] = int(max((v["age"] for v in cands.values()), default=0))
        return spawn

    def note_new_object(self, obj_id):
        self._hist.setdefault(obj_id, {})
        self._stable.setdefault(obj_id, {})

    def reset_object(self, obj_id):
        """Forget accumulated evidence, e.g. after a split changes membership."""
        self._cum.pop(obj_id, None)
        self._cnt.pop(obj_id, None)
        self._tid_index.pop(obj_id, None)

    def drop_tracks(self, obj_id, track_ids):
        h = self._hist.get(obj_id, None)
        if not h:
            return
        for t in np.asarray(track_ids).reshape(-1).tolist():
            h.pop(int(t), None)

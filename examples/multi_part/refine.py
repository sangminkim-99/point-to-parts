"""Step 5: rendering-based refinement, run at a lower rate than the frontend.

Per-part pose is polished against the RGB-D residual of that part's Gaussians,
labels move to whichever part explains them over several frames, and the joint
parameters are refined under the same loss.
"""
import numpy as np


class RenderRefiner:
    """Holds the multi-frame residual history the label update needs."""

    def __init__(self, model, r_max=0.05, hist=8, margin=1.5, votes=3):
        self.m = model
        self.r_max, self.hist = r_max, hist
        self.margin, self.votes = margin, votes
        self._acc = None          # (n_parts, n_gauss) running mean residual
        self._seen = None
        self._votes = None

    # ---- 5a: pose ----
    def refine_poses(self, parts, depth, mask, K, H, W, huber=0.02, iters=8):
        """Projective ICP of each part's own Gaussians onto the observed depth."""
        a = self.m.assign
        out = []
        for j, p in enumerate(parts):
            w = self.m.weights(j)
            if w.sum() < 20 or p.pose is None:
                out.append(None)
                continue
            T = a.refine_pose(p.pose, w, K, H, W, depth, iters=iters,
                              huber=huber, obs_mask=mask)
            e0 = a.fit_energy(p.pose, w, K, H, W, depth, obs_mask=mask,
                              r_max=self.r_max)
            e1 = a.fit_energy(T, w, K, H, W, depth, obs_mask=mask,
                              r_max=self.r_max)
            out.append((T, e1) if e1 < e0 else (p.pose, e0))
        return out

    # ---- 5b: labels, over several frames ----
    def update_labels(self, parts, depth, mask, K, H, W):
        """A gaussian moves to the part that explains it better across frames."""
        a = self.m.assign
        poses = [p.pose for p in parts]
        R = a.residuals(poses, K, H, W, depth, obs_mask=mask)
        R = R.detach().cpu().numpy()
        n = min(R.shape[1], len(self.m.labels))
        R = R[:, :n]
        seen = R < 1e5
        if self._acc is None or self._acc.shape != R.shape:
            self._acc = np.zeros_like(R)
            self._seen = np.zeros_like(R)
            self._votes = np.zeros(n, np.int16)
        k = 1.0 / self.hist
        self._acc = (1 - k) * self._acc + k * np.where(seen, R, self._acc)
        self._seen = (1 - k) * self._seen + k * seen
        good = self._seen > 0.3
        A = np.where(good, self._acc, np.inf)
        best = A.argmin(axis=0)
        bv = A.min(axis=0)
        # defensive: a stale label must never index past the parts
        cur = np.clip(self.m.labels[:n], -1, A.shape[0] - 1)
        here = np.where(cur >= 0, A[np.clip(cur, 0, None), np.arange(n)], np.inf)
        want = (cur >= 0) & (best != cur) & np.isfinite(bv) \
            & (here > bv * self.margin)
        self._votes = np.where(want, self._votes + 1, 0)
        go = np.where(self._votes >= self.votes)[0]
        if go.size:
            self.m.labels[go] = best[go]
            self._votes[go] = 0
        return int(go.size)

    # ---- 5c: joint parameters, under the same loss ----
    def refine_joint(self, parts, j, depth, mask, K, H, W, span=41):
        """Sweep the joint value and keep the pose the render likes best."""
        p = parts[j]
        if p.joint is None or p.joint.kind is None or p.parent >= len(parts):
            return None
        Tp = parts[p.parent].pose
        if Tp is None:
            return None
        w = self.m.weights(j)
        if w.sum() < 20:
            return None
        vs = [p.joint.value_of(A) for A in p.joint.A] or [0.0]
        rng = max(max(vs) - min(vs), 0.1)
        grid = vs[-1] + np.linspace(-1.0, 1.0, span) * rng
        Ts = [Tp @ p.joint.at(float(v)) for v in grid]
        e = self.m.assign.fit_energy_batch(Ts, w, K, H, W, depth, obs_mask=mask,
                                           r_max=self.r_max)
        i = int(np.argmin(e))
        return Ts[i], float(e[i]), float(grid[i])

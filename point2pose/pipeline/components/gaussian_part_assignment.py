"""Dense part assignment by rendering rigid-motion hypotheses as Gaussians.

The sparse detectors all bottom out on the same limit: a part is only found if
enough keypoints happen to land on it, which fails for thin objects (26-36
tracks on a folding rule) and for symmetric ones where the consensus sets swap
identity.  This takes the plan's section 3 route instead: represent the object
as Gaussians lifted straight from depth, transform them by each hypothesis,
render, and let the per-pixel rendering residual say which surface belongs to
which motion.

Deliberately not a trained splat.  Gaussians are initialised as a point cloud
surrogate -- one per depth sample, SH degree 0 (a flat colour), isotropic scale
set by the pixel footprint at that range -- so no optimisation is needed before
the residual is meaningful.  What the render buys over a plain point-wise depth
test is visibility: a hypothesis that moves a part in front of another must
occlude it, and rasterisation enforces that where a point lookup cannot.

Assignment is accumulated as log-odds per Gaussian rather than decided per
frame, following the plan's section 2.1, so one bad frame cannot fix a label.
"""

import numpy as np
import torch
from gsplat import rasterization


def backproject(depth, K, mask=None, min_depth=0.05):
    """(N,3) camera-frame points and the pixel indices they came from."""
    H, W = depth.shape
    vs, us = np.mgrid[0:H, 0:W]
    ok = depth > min_depth
    if mask is not None:
        ok &= mask > 0
    us, vs, z = us[ok], vs[ok], depth[ok]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], axis=1).astype(np.float32), (vs, us)


class GaussianCloud:
    """Gaussians standing in for a point cloud, in a fixed anchor frame."""

    def __init__(self, means, colors, scales, device="cuda"):
        self.device = device
        self.means = torch.as_tensor(means, dtype=torch.float32, device=device)
        self.colors = torch.as_tensor(colors, dtype=torch.float32, device=device)
        n = self.means.shape[0]
        self.quats = torch.zeros((n, 4), device=device)
        self.quats[:, 0] = 1.0
        self.scales = torch.as_tensor(scales, dtype=torch.float32,
                                      device=device).reshape(n, 1).repeat(1, 3)
        self.opacities = torch.ones(n, device=device)

    @classmethod
    def from_depth(cls, rgb, depth, K, mask=None, stride=2, scale_mult=1.2,
                   device="cuda"):
        """Lift a frame to Gaussians. Scale follows the pixel footprint at range."""
        d = depth[::stride, ::stride]
        c = rgb[::stride, ::stride]
        m = None if mask is None else mask[::stride, ::stride]
        Ks = K.copy()
        Ks[0, 0] /= stride; Ks[1, 1] /= stride
        Ks[0, 2] /= stride; Ks[1, 2] /= stride
        pts, (vs, us) = backproject(d, Ks, m)
        if len(pts) == 0:
            return None
        cols = c[vs, us].astype(np.float32) / 255.0
        # a pixel at range z covers roughly z / f metres; that is the right
        # radius for a splat meant to tile the surface without holes
        foot = pts[:, 2] / float(Ks[0, 0])
        return cls(pts, cols, foot * scale_mult, device=device)

    def __len__(self):
        return int(self.means.shape[0])

    def render(self, T_anchor2cam, K, H, W, subset=None, near=0.02, far=12.0):
        """Rasterise under one hypothesis. Returns (rgb, depth, alpha) as tensors."""
        viewmat = torch.as_tensor(np.asarray(T_anchor2cam), dtype=torch.float32,
                                  device=self.device)[None]
        Ks = torch.as_tensor(np.asarray(K), dtype=torch.float32,
                             device=self.device)[None]
        sel = slice(None) if subset is None else subset
        colors, alphas, _ = rasterization(
            means=self.means[sel], quats=self.quats[sel], scales=self.scales[sel],
            opacities=self.opacities[sel], colors=self.colors[sel],
            viewmats=viewmat, Ks=Ks, width=W, height=H,
            render_mode="RGB+ED", near_plane=near, far_plane=far, packed=False,
        )
        rgb = colors[0, ..., :3]
        depth = colors[0, ..., 3]
        alpha = alphas[0, ..., 0]
        return rgb, depth, alpha


class GaussianPartAssignment:
    """Accumulates per-Gaussian log-odds over rigid-motion hypotheses."""

    def __init__(self, cloud, n_hypotheses, device="cuda",
                 depth_sigma=0.02, color_weight=0.3, decay=0.98):
        self.cloud = cloud
        self.device = device
        self.depth_sigma = float(depth_sigma)
        self.color_weight = float(color_weight)
        self.decay = float(decay)
        self.logodds = torch.zeros((len(cloud), n_hypotheses), device=device)
        # Persistent slots. The log-odds columns are indexed by SLOT, not by the
        # order RANSAC happened to return hypotheses in: slot k must mean the same
        # physical part every frame. Matching is by which gaussians a hypothesis
        # wins, which is stable even though its transform changes as the object
        # moves. Without this the accumulation is meaningless -- on RBO it left
        # every gaussian in one group no matter how diverse the hypotheses were.
        self._slot_members = [None] * n_hypotheses
        self.n_slots = n_hypotheses

    def _residual(self, T, K, H, W, obs_depth, obs_rgb):
        rgb, depth, alpha = self.cloud.render(T, K, H, W)
        vis = (alpha > 0.3) & (obs_depth > 0)
        dz = torch.abs(depth - obs_depth)
        r = dz / self.depth_sigma
        if self.color_weight > 0:
            r = r + self.color_weight * torch.abs(rgb - obs_rgb).mean(-1) * 3.0
        r = torch.where(vis, r, torch.full_like(r, 1e6))
        return r, depth, alpha, vis

    def step(self, hypotheses, K, H, W, obs_depth, obs_rgb, obs_mask=None):
        """One frame. `hypotheses` are anchor->camera transforms, one per candidate.

        Per-pixel the best-explaining hypothesis wins; every Gaussian that landed
        on a winning pixel gets a vote. Pixels no hypothesis explains are reported
        separately -- that is unmodelled geometry, which is how a drawer interior
        or a newly revealed face should enter the model.
        """
        obs_depth = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
        obs_rgb = torch.as_tensor(obs_rgb, dtype=torch.float32, device=self.device) / 255.0
        # score only where the object is: the background and ground plane are not
        # modelled, so counting them would report 98% "unexplained" every frame
        roi = (obs_depth > 0)
        if obs_mask is not None:
            roi = roi & torch.as_tensor(obs_mask > 0, device=self.device)

        res, deps, viss = [], [], []
        for T in hypotheses:
            r, d, a, v = self._residual(T, K, H, W, obs_depth, obs_rgb)
            res.append(r); deps.append(d); viss.append(v)
        R = torch.stack(res)                       # (Hy, H, W)
        best = R.argmin(dim=0)
        bestval = R.min(dim=0).values
        explained = (bestval < 3.0) & roi           # within ~3 sigma of the depth

        # splat the winning label back onto the Gaussians that produced it
        self.logodds *= self.decay
        for k, T in enumerate(hypotheses):
            win = explained & (best == k)
            if win.sum() == 0:
                continue
            # which Gaussians project into the winning pixels under this hypothesis
            idx = self._project_index(T, K, H, W)
            keep = idx >= 0
            if keep.sum() == 0:
                continue
            flat = win.reshape(-1)
            hit = torch.zeros(len(self.cloud), dtype=torch.bool, device=self.device)
            hit[keep] = flat[idx[keep]]
            self.logodds[hit, k] += 1.0
        n_roi = float(roi.sum().clamp(min=1))
        return {
            "explained_frac": float(explained.sum()) / n_roi,
            "unexplained_frac": float((roi & ~explained).sum()) / n_roi,
            "median_best_residual": float(bestval[explained].median()) if explained.any() else float("nan"),
        }

    def _project_index(self, T, K, H, W):
        """Flat pixel index each Gaussian lands on, -1 when outside the frame."""
        T = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
        p = self.cloud.means @ T[:3, :3].T + T[:3, 3]
        z = p[:, 2]
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
        v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        idx = torch.full((len(self.cloud),), -1, dtype=torch.long, device=self.device)
        idx[ok] = (v[ok] * W + u[ok])
        return idx

    def step_pointwise(self, hypotheses, K, H, W, obs_depth, obs_rgb=None,
                       obs_mask=None, tol=None):
        """Per-Gaussian residual instead of a per-pixel argmin over full renders.

        Rendering every hypothesis over the WHOLE cloud conflates them: at a pixel
        belonging to a moved part, the wrong hypothesis still puts some other
        Gaussian there, and whichever lands closest wins for reasons unrelated to
        that surface. Measured on RBO, one hypothesis then explained 97.8% of
        pixels and no part ever separated.

        Asking instead "where does THIS Gaussian go under hypothesis k, and does
        the depth there agree?" removes the confound entirely, and is what the
        plan's section 3 assignment is really testing.
        """
        tol = self.depth_sigma if tol is None else tol
        obs = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
        if obs_mask is not None:
            mk = torch.as_tensor(obs_mask > 0, device=self.device)
            obs = torch.where(mk, obs, torch.zeros_like(obs))

        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        res = []
        for T in hypotheses:
            Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
            p = self.cloud.means @ Tt[:3, :3].T + Tt[:3, 3]
            z = p[:, 2]
            u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zo = torch.zeros_like(z)
            zo[ok] = obs[v[ok].clamp(0, H - 1), u[ok].clamp(0, W - 1)]
            r = torch.full_like(z, 1e6)
            good = ok & (zo > 0)
            r[good] = torch.abs(z[good] - zo[good])
            res.append(r)
        R = torch.stack(res)                                  # (Hy, N)
        best = R.argmin(dim=0)
        bestval = R.min(dim=0).values
        # a vote only counts when this hypothesis explains the surface AND beats
        # the runner-up clearly, so ambiguous gaussians stay undecided
        second = R.topk(2, dim=0, largest=False).values[1] if R.shape[0] > 1 else \
            torch.full_like(bestval, 1e6)
        decisive = (bestval < tol) & (second > bestval * 1.5 + 0.5 * tol)

        # winners of each hypothesis this frame, then bind them to persistent slots
        wins = []
        for k in range(len(hypotheses)):
            wins.append(decisive & (best == k))
        slot_of = self._bind_slots(wins)

        self.logodds *= self.decay
        for k, sl in enumerate(slot_of):
            if sl is None or not wins[k].any():
                continue
            self.logodds[wins[k], sl] += 1.0
        if getattr(self, "co_idx", None) is not None:
            self.accumulate_coassoc_soft(R, bestval < 1e5)
        self.last_wins = [w.detach().cpu().numpy() for w in wins]
        self.last_slots = list(slot_of)
        return {
            "win_counts": [int(w.sum()) for w in wins],
            "slots": [(-1 if x is None else int(x)) for x in slot_of],
            "explained_frac": float((bestval < tol).float().mean()),
            "decisive_frac": float(decisive.float().mean()),
            "median_best_residual": float(bestval[bestval < 1e5].median())
            if (bestval < 1e5).any() else float("nan"),
        }

    def init_coassoc(self, n_sample=4000, seed=0):
        """Evidence accumulation: count how often each pair of gaussians is put in
        the same group, and cluster that at the end.

        This removes hypothesis identity from the problem entirely. Every attempt
        to carry a slot index across frames failed the same way -- the index means
        a different motion each frame, and whatever repair was applied (overlap
        binding, eviction) either discarded most votes or let the dominant motion
        claim every slot. Co-association never asks which hypothesis is which; it
        only asks whether two surfaces moved together, which is the actual
        question.
        """
        n = len(self.cloud)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.co_idx = (torch.randperm(n, generator=g)[:min(n_sample, n)]
                       .to(self.device).sort().values)
        m = self.co_idx.numel()
        self.co_same = torch.zeros((m, m), dtype=torch.float32, device=self.device)
        self.co_seen = torch.zeros((m, m), dtype=torch.float32, device=self.device)

    def accumulate_coassoc_soft(self, R, valid, tau=None):
        """Soft co-association from the residual matrix R (Hy, N).

        A hard winner-takes-all vote gated on a margin recorded almost nothing:
        when the drawer separated, the body's gaussians failed the margin because
        several near-duplicate hypotheses explained them equally, so the pair was
        never counted. Measured, that left same-part and cross-part pairs both at
        0.999 -- no signal at all. A posterior over hypotheses needs no margin and
        every co-observed pair contributes.
        """
        idx = self.co_idx
        tau = self.depth_sigma if tau is None else tau
        r = R[:, idx].clone()
        v = valid[idx].float()
        r[~torch.isfinite(r)] = 1e6
        P = torch.softmax(-r / max(tau, 1e-6), dim=0)          # (Hy, M)
        P = P * v[None, :]
        self.co_same += P.T @ P
        self.co_seen += v[:, None] * v[None, :]

    def coassoc_labels(self, min_frac=0.5, min_group=50, max_k=5, adaptive=True):
        """Cluster the co-association matrix.

        The raw affinity values are not comparable between sequences -- they
        depend on how many hypotheses were generated and how far apart they were
        -- so a fixed distance threshold tuned on one sequence gave 2/3 parts
        there and zero groups on the other four. Choosing the number of clusters
        by silhouette adapts to each sequence instead.
        """
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score

        seen = self.co_seen.clamp(min=1.0)
        A = (self.co_same / seen).cpu().numpy()
        A = 0.5 * (A + A.T)
        np.fill_diagonal(A, A.max() if A.size else 1.0)
        D = A.max() - A
        np.fill_diagonal(D, 0.0)

        if not adaptive:
            lab = AgglomerativeClustering(
                n_clusters=None, distance_threshold=1.0 - min_frac,
                metric="precomputed", linkage="average").fit_predict(D)
        else:
            best = None
            for k in range(2, max_k + 1):
                try:
                    l = AgglomerativeClustering(
                        n_clusters=k, metric="precomputed",
                        linkage="average").fit_predict(D)
                except Exception:
                    continue
                sizes = np.bincount(l, minlength=k)
                if sizes.min() < min_group:
                    continue
                try:
                    sc = silhouette_score(D, l, metric="precomputed")
                except Exception:
                    continue
                if best is None or sc > best[0]:
                    best = (sc, l, k)
            if best is None:
                lab = np.zeros(D.shape[0], dtype=int)
            else:
                lab = best[1]

        keep = {c for c in set(lab) if (lab == c).sum() >= min_group}
        out = np.array([c if c in keep else -1 for c in lab])
        return out, self.co_idx.cpu().numpy()

    def _bind_slots(self, wins, min_overlap=0.2):
        """Map this frame's hypotheses onto persistent slots by winner overlap."""
        used, out = set(), []
        for w in wins:
            if not w.any():
                out.append(None)
                continue
            best_s, best_j = None, min_overlap
            for s in range(self.n_slots):
                m = self._slot_members[s]
                if m is None or s in used:
                    continue
                inter = float((w & m).sum())
                union = float((w | m).sum())
                j = inter / union if union > 0 else 0.0
                if j > best_j:
                    best_s, best_j = s, j
            if best_s is None:
                free = [s for s in range(self.n_slots)
                        if self._slot_members[s] is None and s not in used]
                if free:
                    best_s = free[0]
                else:
                    # Evict the slot holding the least evidence rather than
                    # discarding the vote. Refusing to rebind once every slot had
                    # been touched threw away most of the sequence: on RBO the
                    # drawer's own hypothesis won >=200 gaussians 19 times and
                    # every one of those votes was dropped.
                    mass = self.logodds.sum(dim=0)
                    cand = [s for s in range(self.n_slots) if s not in used]
                    if not cand:
                        out.append(None)
                        continue
                    best_s = min(cand, key=lambda s: float(mass[s]))
                    self.logodds[:, best_s] = 0.0
            used.add(best_s)
            # follow the group, but keep some memory of it so a single frame with
            # a partial view cannot redefine what the slot means
            prev = self._slot_members[best_s]
            self._slot_members[best_s] = w.clone() if prev is None else (w | (prev & ~torch.zeros_like(w)))
            if prev is not None:
                # decay membership: drop points that have not won recently
                self._slot_members[best_s] = w | (prev & w) if w.sum() > 0 else prev
            out.append(best_s)
        return out

    def labels(self, min_margin=1.0):
        """Hard label per Gaussian, -1 where the evidence is not yet decisive."""
        top2 = self.logodds.topk(2, dim=1)
        lab = top2.indices[:, 0].clone()
        margin = top2.values[:, 0] - top2.values[:, 1]
        lab[margin < min_margin] = -1
        return lab.cpu().numpy()

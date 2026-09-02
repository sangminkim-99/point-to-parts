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
        self.coassoc_weight = "none"
        # ceiling on a winner set, as a fraction of the cloud. 0.5 assumes at
        # least three parts: for a TWO-part object one part legitimately covers
        # about half the cloud, and a 0.5 ceiling throws away exactly the sets
        # that matter. Measured on RBO pliers, where the median recorded set is
        # 38.9% of the cloud against ~2% on a three-drawer cabinet.
        self.winset_max_frac = 0.5

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
            self.accumulate_coassoc_soft(R, bestval < 1e5,
                                         weight_mode=self.coassoc_weight)
        self.record_winsets(wins, max_frac=self.winset_max_frac)
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

    # ------------------------------------------------------------------ #
    #  M-step: refine each hypothesis against the dense model
    # ------------------------------------------------------------------ #

    def refine_pose(self, T, weights, K, H, W, obs_depth, iters=6, huber=0.02,
                    obs_mask=None, visible=None):
        """Projective ICP of the gaussians onto the observed depth.

        The plan's section 3.1 EM loop needs this M-step, and every RBO failure
        traced to hypothesis quality rather than assignment: a single sparse
        RANSAC hypothesis matched all three parts to within 6-25 mm. Refining a
        group's transform against the dense surface it is supposed to explain
        attacks the problem where it actually is.

        `weights` is a soft membership over gaussians (N,) in [0,1].
        """
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
        if obs_mask is not None:
            # Without this the ground plane is a valid ICP target and drags the
            # part onto it: the moving scissor blade came out 300 mm off while
            # the static part was exact.
            mk = torch.as_tensor(obs_mask > 0, device=self.device)
            obs = torch.where(mk, obs, torch.zeros_like(obs))
        P0 = self.cloud.means
        T = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device).clone()
        w0 = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        if visible is not None:
            # computed once from the starting pose: within one ICP the pose
            # moves little, and a render per iteration is not worth 0.25 ms each
            w0 = w0 * visible[:w0.shape[0]].float()
        if float(w0.sum()) < 10:
            return T.cpu().numpy()

        for _ in range(iters):
            p = P0 @ T[:3, :3].T + T[:3, 3]
            z = p[:, 2]
            u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (w0 > 0.05)
            if int(ok.sum()) < 10:
                break
            zo = obs[v[ok].clamp(0, H - 1), u[ok].clamp(0, W - 1)]
            good = zo > 0
            if int(good.sum()) < 10:
                break
            idx = torch.where(ok)[0][good]
            src = p[idx]
            # the observed surface point along the same ray
            zt = zo[good]
            tgt = torch.stack([
                (u[idx].float() - Kt[0, 2]) * zt / Kt[0, 0],
                (v[idx].float() - Kt[1, 2]) * zt / Kt[1, 1],
                zt], dim=1)

            r = torch.linalg.norm(tgt - src, dim=1)
            wt = w0[idx] * torch.clamp(huber / torch.clamp(r, min=1e-6), max=1.0)
            if float(wt.sum()) < 1e-6:
                break
            wn = (wt / wt.sum())[:, None]
            cs = (src * wn).sum(0)
            ct = (tgt * wn).sum(0)
            Hm = ((src - cs) * wn).T @ (tgt - ct)
            try:
                U, _, Vt = torch.linalg.svd(Hm)
            except Exception:
                break
            d = torch.sign(torch.det(Vt.T @ U.T))
            Rc = Vt.T @ torch.diag(torch.tensor([1.0, 1.0, float(d)],
                                                device=self.device)) @ U.T
            dT = torch.eye(4, device=self.device)
            dT[:3, :3] = Rc
            dT[:3, 3] = ct - Rc @ cs
            T = dT @ T
        return T.cpu().numpy()

    def fit_error(self, T, weights, K, H, W, obs_depth, obs_mask=None):
        """Median depth residual of a weighted gaussian set under transform T."""
        R = self.residuals([T], K, H, W, obs_depth, obs_mask=obs_mask)[0]
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        ok = (R < 1e5) & (w > 0.05)
        if int(ok.sum()) < 10:
            return float("inf")
        return float(R[ok].median())

    def self_visible(self, T, weights, K, H, W, tol=0.006, alpha_thr=0.3):
        """Which of a part's gaussians its own front face does not hide.

        The point-wise residual has no z-buffer, so a gaussian on the back of a
        part is compared against the front face's depth and charged for the
        thickness. Measured: a model 10 mm behind the surface reads 11.2 mm
        point-wise and 7.9 mm rasterised.
        """
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        sel = w[:len(self.cloud)] > 0.05
        out = torch.zeros(len(self.cloud), dtype=torch.bool, device=self.device)
        if int(sel.sum()) < 10:
            return out
        _, dep_r, alpha = self.cloud.render(T, K, H, W, subset=sel)
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
        p = self.cloud.means[sel] @ Tt[:3, :3].T + Tt[:3, 3]
        z = p[:, 2]
        u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
        v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
        # in front of, or on, the surface this part renders at that pixel
        vis = ok & (alpha[vc, uc] > alpha_thr) & (z <= dep_r[vc, uc] + tol)
        out[torch.where(sel)[0]] = vis
        return out

    def fit_energy(self, T, weights, K, H, W, obs_depth, obs_mask=None,
                   r_max=0.05, obs_rgb=None, color_w=0.0, visible=None):
        """Truncated residual energy over ALL of a part's gaussians.

        `fit_error` is a median over the gaussians that happened to land on
        valid depth, so a pose that slides most of the part off the observed
        surface and parks a sliver of it on a neighbouring part scores
        beautifully. That is exactly how the laptop lid -- correctly segmented
        at 94% -- ended up 800 mm and 80 deg away: the candidate that abandoned
        the part won the comparison. Counting a miss as a full `r_max` makes
        losing support as expensive as fitting badly.

        `color_w` adds the gaussians' own colour against the observed pixel.
        Depth alone cannot orient a flat part: a laptop lid is a slab, and every
        in-plane rotation of it fits the depth equally well, which is why the
        translation came out within 30 mm while the rotation stayed 8-25 deg off.
        """
        e = self.fit_energy_batch([T], weights, K, H, W, obs_depth,
                                  obs_mask=obs_mask, r_max=r_max,
                                  obs_rgb=obs_rgb, color_w=color_w,
                                  visible=visible)
        return float(e[0])

    def fit_energy_batch(self, Ts, weights, K, H, W, obs_depth, obs_mask=None,
                         r_max=0.05, chunk=64, obs_rgb=None, color_w=0.0,
                         visible=None):
        """`fit_energy` for many poses at once, over a part's members only.

        A joint reduces the pose to a scalar, and a scalar can be scanned
        exhaustively -- but only if scoring a sample is cheap. Restricting to the
        part's own gaussians and batching over poses makes a 120-sample sweep of
        the whole joint range cost about as much as a handful of ICP starts.
        """
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
        if obs_mask is not None:
            mk = torch.as_tensor(obs_mask > 0, device=self.device)
            obs = torch.where(mk, obs, torch.zeros_like(obs))
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        sel = w[:len(self.cloud)] > 0.05
        if visible is not None:
            # drop only gaussians the part's own front face hides; ones that
            # missed the surface entirely still count, as a full r_max
            sel = sel & visible[:len(self.cloud)]
        if int(sel.sum()) < 10:
            return np.full(len(Ts), float("inf"))
        P = self.cloud.means[sel]
        C = self.cloud.colors[sel] if color_w > 0 else None
        rgbt = None
        if color_w > 0 and obs_rgb is not None:
            rgbt = torch.as_tensor(obs_rgb, dtype=torch.float32,
                                   device=self.device) / 255.0
        out = []
        for i in range(0, len(Ts), chunk):
            Tb = torch.as_tensor(np.stack([np.asarray(t) for t in Ts[i:i + chunk]]),
                                 dtype=torch.float32, device=self.device)
            p = torch.einsum("gij,nj->gni", Tb[:, :3, :3], P) + Tb[:, None, :3, 3]
            z = p[..., 2]
            u = (p[..., 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (p[..., 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            uc, vc = u.clamp(0, W - 1), v.clamp(0, H - 1)
            zo = obs[vc, uc]
            r = torch.full_like(z, r_max)
            g = ok & (zo > 0)
            r[g] = torch.clamp(torch.abs(z[g] - zo[g]), max=r_max)
            if rgbt is not None:
                dc = torch.abs(rgbt[vc, uc] - C[None]).mean(-1)
                r = r + color_w * r_max * torch.clamp(dc / 0.25, max=1.0) * g
            out.append(r.mean(dim=1))
        return torch.cat(out).cpu().numpy()

    def occupancy(self, T, weights, K, H, W, dilate=1, alpha_thr=0.3):
        """Boolean image of the pixels a part covers under transform T.

        Rasterised: splatting gaussian centres and dilating leaves holes between
        samples, which the mutual-exclusion mask then hands to another part.
        """
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        sel = w[:len(self.cloud)] > 0.05
        if int(sel.sum()) >= 10:
            _, _, alpha = self.cloud.render(T, K, H, W, subset=sel)
            img = alpha > alpha_thr
            if dilate > 0:
                f = torch.nn.functional.max_pool2d(
                    img[None, None].float(), 2 * dilate + 1, stride=1, padding=dilate)
                img = f[0, 0] > 0
            return img.cpu().numpy()
        return self._occupancy_points(T, weights, K, H, W, dilate)

    def _occupancy_points(self, T, weights, K, H, W, dilate=1):
        """Fallback for a part too small to rasterise."""
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
        w = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        p = self.cloud.means @ Tt[:3, :3].T + Tt[:3, 3]
        z = p[:, 2]
        u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
        v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
        ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (w > 0.05)
        img = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        if int(ok.sum()) == 0:
            return img.cpu().numpy()
        img[v[ok], u[ok]] = True
        if dilate > 0:
            f = torch.nn.functional.max_pool2d(
                img[None, None].float(), 2 * dilate + 1, stride=1, padding=dilate)
            img = f[0, 0] > 0
        return img.cpu().numpy()

    def soft_membership(self, R, tau=None):
        """Posterior over hypotheses per gaussian, (Hy, N)."""
        tau = self.depth_sigma if tau is None else tau
        r = R.clone()
        r[~torch.isfinite(r)] = 1e6
        return torch.softmax(-r / max(tau, 1e-6), dim=0)

    def residuals(self, hypotheses, K, H, W, obs_depth, obs_mask=None):
        """Per-gaussian depth residual under each hypothesis, (Hy, N)."""
        obs = torch.as_tensor(obs_depth, dtype=torch.float32, device=self.device)
        if obs_mask is not None:
            mk = torch.as_tensor(obs_mask > 0, device=self.device)
            obs = torch.where(mk, obs, torch.zeros_like(obs))
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        out = []
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
            g = ok & (zo > 0)
            r[g] = torch.abs(z[g] - zo[g])
            out.append(r)
        return torch.stack(out)

    # ------------------------------------------------------------------ #
    #  Growth: surfaces that appear later must enter the model
    # ------------------------------------------------------------------ #

    def grow(self, rgb, depth, K, mask, hypotheses, T_anchor, tol=0.03,
             stride=3, max_new=4000):
        """Add gaussians for object pixels no hypothesis explains.

        Without this the model is whatever the first frame happened to see, and a
        drawer interior or a newly revealed face can never be represented -- the
        plan calls these out specifically as how unmodelled geometry should enter.
        New points are anchored through the dominant hypothesis; which part they
        belong to is left to the usual accumulation.
        """
        H, W = depth.shape
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        covered = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        obs = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        for T in hypotheses:
            Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
            p = self.cloud.means @ Tt[:3, :3].T + Tt[:3, 3]
            z = p[:, 2]
            u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            uu, vv, zz = u[ok], v[ok], z[ok]
            zo = obs[vv, uu]
            hit = (zo > 0) & (torch.abs(zz - zo) < tol)
            covered[vv[hit], uu[hit]] = True

        m = torch.as_tensor(mask > 0, device=self.device) & (obs > 0) & ~covered
        sel = torch.zeros_like(m)
        sel[::stride, ::stride] = True
        m = m & sel
        n_new = int(m.sum())
        if n_new == 0:
            return 0
        vs, us = torch.where(m)
        if n_new > max_new:
            pick = torch.randperm(n_new, device=self.device)[:max_new]
            vs, us = vs[pick], us[pick]
            n_new = max_new
        z = obs[vs, us]
        pc = torch.stack([(us.float() - Kt[0, 2]) * z / Kt[0, 0],
                          (vs.float() - Kt[1, 2]) * z / Kt[1, 1], z], dim=1)
        Ta = torch.as_tensor(np.asarray(T_anchor), dtype=torch.float32,
                             device=self.device)
        # anchor-frame position: R^T (p_cam - t), written for row vectors
        pa = (pc - Ta[:3, 3]) @ Ta[:3, :3]
        cols = torch.as_tensor(rgb[vs.cpu().numpy(), us.cpu().numpy()],
                               dtype=torch.float32, device=self.device) / 255.0
        foot = (z / float(Kt[0, 0])) * 1.2

        c = self.cloud
        c.means = torch.cat([c.means, pa], 0)
        c.colors = torch.cat([c.colors, cols], 0)
        c.scales = torch.cat([c.scales, foot[:, None].repeat(1, 3)], 0)
        c.opacities = torch.cat([c.opacities, torch.ones(n_new, device=self.device)], 0)
        q = torch.zeros((n_new, 4), device=self.device); q[:, 0] = 1.0
        c.quats = torch.cat([c.quats, q], 0)
        self.logodds = torch.cat(
            [self.logodds, torch.zeros((n_new, self.logodds.shape[1]),
                                       device=self.device)], 0)
        return n_new

    def grow_parts(self, rgb, depth, K, mask, poses, weights, tol=0.02,
                   stride=3, max_new=1500, max_total=300000, grow_ok=None):
        """Attach newly revealed surface to the part it is adjacent to.

        `grow` anchors new points through the *first* hypothesis, which is fine
        for reporting unmodelled geometry but useless for tracking: a laptop lid
        rotating 130 degrees turns its back face to the camera, and that face is
        not in the model at all. The lid's own gaussians then sit roughly one lid
        thickness behind the observed surface, so even the ground-truth pose only
        reaches a residual of ~10 mm and projective ICP has nothing to lock onto.
        Once the parts are known, an unexplained pixel can be given to the part it
        borders in the image and anchored through that part's current pose, so the
        model of each part fills in as the object articulates.

        Returns (n_new, owner) with `owner` the part index of each new gaussian.
        """
        import cv2
        H, W = depth.shape
        if len(self.cloud) >= max_total:
            return 0, None
        Kt = torch.as_tensor(np.asarray(K), dtype=torch.float32, device=self.device)
        obs = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        lab = torch.full((H, W), -1, dtype=torch.long, device=self.device)
        res = torch.full((H, W), 1e9, dtype=torch.float32, device=self.device)
        for k, (T, w) in enumerate(zip(poses, weights)):
            if T is None:
                continue
            wt = torch.as_tensor(w, dtype=torch.float32, device=self.device)
            sel = wt[:len(self.cloud)] > 0.5
            if int(sel.sum()) < 10:
                continue
            Tt = torch.as_tensor(np.asarray(T), dtype=torch.float32, device=self.device)
            p = self.cloud.means[sel] @ Tt[:3, :3].T + Tt[:3, 3]
            z = p[:, 2]
            u = (p[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (p[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not ok.any():
                continue
            uu, vv, zz = u[ok], v[ok], z[ok]
            zo = obs[vv, uu]
            d = torch.abs(zz - zo)
            hit = (zo > 0) & (d < tol)
            uu, vv, d = uu[hit], vv[hit], d[hit]
            better = d < res[vv, uu]
            res[vv[better], uu[better]] = d[better]
            lab[vv[better], uu[better]] = k

        cov = (lab >= 0).cpu().numpy()
        if not cov.any():
            return 0, None
        m = (torch.as_tensor(mask > 0, device=self.device) & (obs > 0)
             & (lab < 0))
        sel = torch.zeros_like(m)
        sel[::stride, ::stride] = True
        m = m & sel
        n_new = int(m.sum())
        if n_new == 0:
            return 0, None

        # nearest covered pixel decides which part a new surface joins: a face
        # that has just come into view borders the part it belongs to
        _, near = cv2.distanceTransformWithLabels(
            (~cov).astype(np.uint8), cv2.DIST_L2, 3,
            labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.where(cov)
        # DIST_LABEL_PIXEL numbers the zero pixels 1..n in raster order
        order = np.argsort(ys * W + xs)
        ys, xs = ys[order], xs[order]
        labn = lab.cpu().numpy()

        vs, us = torch.where(m)
        if n_new > max_new:
            pick = torch.randperm(n_new, device=self.device)[:max_new]
            vs, us = vs[pick], us[pick]
            n_new = max_new
        vn, un = vs.cpu().numpy(), us.cpu().numpy()
        li = near[vn, un] - 1
        li = np.clip(li, 0, len(ys) - 1)
        owner = labn[ys[li], xs[li]]

        # Only parts whose pose is currently trusted may absorb new surface.
        # Growing through a pose that has already slipped writes the error into
        # the model permanently -- measured on the laptop lid, whose residual at
        # the ground-truth pose doubled over the sequence when growth was
        # ungated.
        if grow_ok is not None:
            keep = np.array([bool(grow_ok[o]) if 0 <= o < len(grow_ok) else False
                             for o in owner])
            if not keep.any():
                return 0, None
            owner = owner[keep]
            kt = torch.as_tensor(keep, device=self.device)
            vs, us = vs[kt], us[kt]
            vn, un = vs.cpu().numpy(), us.cpu().numpy()
            n_new = int(keep.sum())

        z = obs[vs, us]
        pc = torch.stack([(us.float() - Kt[0, 2]) * z / Kt[0, 0],
                          (vs.float() - Kt[1, 2]) * z / Kt[1, 1], z], dim=1)
        cols = torch.as_tensor(rgb[vn, un], dtype=torch.float32,
                               device=self.device) / 255.0
        foot = (z / float(Kt[0, 0])) * 1.2

        pa = torch.zeros_like(pc)
        for k, T in enumerate(poses):
            if T is None:
                continue
            g = torch.as_tensor(owner == k, device=self.device)
            if not g.any():
                continue
            Ta = torch.as_tensor(np.asarray(T), dtype=torch.float32,
                                 device=self.device)
            pa[g] = ((pc[g] - Ta[:3, 3]) @ Ta[:3, :3]).float()

        c = self.cloud
        c.means = torch.cat([c.means, pa], 0)
        c.colors = torch.cat([c.colors, cols], 0)
        c.scales = torch.cat([c.scales, foot[:, None].repeat(1, 3)], 0)
        c.opacities = torch.cat([c.opacities, torch.ones(n_new, device=self.device)], 0)
        q = torch.zeros((n_new, 4), device=self.device); q[:, 0] = 1.0
        c.quats = torch.cat([c.quats, q], 0)
        return n_new, owner

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

    def accumulate_coassoc_soft(self, R, valid, tau=None, weight_mode="none"):
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
        # A frame in which one hypothesis explains nearly every gaussian carries
        # no information about which surfaces move together -- it pushes EVERY
        # pair towards "same" and so shrinks the gap the clustering depends on.
        # Measured on RBO cabinet01: only 10% of gaussians are decisive in the
        # median frame, and the resulting same-part / cross-part affinity gap is
        # 0.035. Weighting each frame by how far its posterior is from a single
        # hypothesis owning everything lets the informative frames dominate.
        wf = 1.0
        if weight_mode != "none":
            n_v = float(v.sum())
            if n_v < 1:
                return
            if weight_mode == "decisive":
                # how far the mean posterior is from one hypothesis owning
                # everything. Measured on RBO this barely varies frame to frame
                # -- the posteriors are soft, so the weight is nearly constant
                # and changes nothing.
                wf = max(0.0, 1.0 - float((P.sum(dim=1) / n_v).max()))
            elif weight_mode == "disagree":
                # the sharper question: what fraction of gaussians do NOT pick
                # the majority hypothesis. A frame where every surface picks the
                # same motion says nothing about which surfaces move together.
                am = P.argmax(dim=0)[v > 0]
                if am.numel() < 1:
                    return
                maj = float(torch.bincount(am, minlength=P.shape[0]).max())
                wf = max(0.0, 1.0 - maj / float(am.numel()))
            if wf <= 1e-3:
                return
        self.co_same += wf * (P.T @ P)
        self.co_seen += wf * (v[:, None] * v[None, :])

    # ------------------------------------------------------------------ #
    #  Grouping by recurring winner sets
    # ------------------------------------------------------------------ #
    def record_winsets(self, wins, min_group=None, max_frac=0.5,
                       min_frac=0.02, min_abs=25):
        """Keep every decisive winner set that is substantial but not the whole
        object.

        Co-association averages over all frames, and on RBO that is fatal: only
        about a tenth of gaussians are decisive in the median frame, so the many
        frames in which one hypothesis explains everything push every pair
        towards "same" and leave a same/cross affinity gap of 0.035. Yet the
        evidence IS there -- cabinet01 has frames where a hypothesis wins an
        rb1-dominant set at 92-99% purity. A set like that recurs, frame after
        frame, and is similar to itself; averaging is what destroys it. So keep
        the sets and cluster the SETS instead of the pairs.

        Sets covering more than `max_frac` of the object are the "everything
        moved together" case and carry no separation information.
        """
        if not hasattr(self, "winsets"):
            self.winsets = []
        n = len(self.cloud)
        # The floor has to scale with the cloud, not be an absolute count. At a
        # fixed 100 gaussians the admissible window is 0.6-50% of a 17k-gaussian
        # RBO cabinet but 15-50% of a 648-gaussian pair of pliers -- nearly shut.
        # Measured: on the five RBO pliers sequences winner-set grouping was
        # never once selected, while on the folding rules (1.3k-2k gaussians) it
        # won every sequence it solved.
        if min_group is None:
            min_group = max(min_abs, int(min_frac * n))
        for w in wins:
            c = int(w.sum())
            if c < min_group or c > max_frac * n:
                continue
            self.winsets.append(
                torch.where(w)[0].detach().cpu().numpy().astype(np.int32))

    def winset_labels(self, min_members=3, thresh=0.5, min_group=None,
                      core_frac=0.5, min_frac=0.02, min_abs=25):
        """Cluster the recorded winner sets by Jaccard overlap; each cluster's
        core is a part. Returns (labels over all gaussians, indices kept)."""
        sets = getattr(self, "winsets", [])
        n = len(self.cloud)
        if min_group is None:
            min_group = max(min_abs, int(min_frac * n))
        if len(sets) < 2 * min_members:
            return np.full(n, -1, dtype=int), np.arange(n)
        M = np.zeros((len(sets), n), dtype=bool)
        for i, idx in enumerate(sets):
            idx = idx[idx < n]
            M[i, idx] = True
        Mf = M.astype(np.float32)
        inter = Mf @ Mf.T
        sz = Mf.sum(1)
        union = sz[:, None] + sz[None, :] - inter
        J = inter / np.maximum(union, 1.0)
        D = np.clip(1.0 - J, 0.0, 1.0)
        np.fill_diagonal(D, 0.0)

        from sklearn.cluster import AgglomerativeClustering
        cl = AgglomerativeClustering(n_clusters=None, metric="precomputed",
                                     linkage="average",
                                     distance_threshold=1.0 - thresh).fit(D)
        labels = np.full(n, -1, dtype=int)
        score = np.zeros(n, dtype=np.float32)
        g = 0
        for c in np.unique(cl.labels_):
            rows = np.where(cl.labels_ == c)[0]
            if rows.size < min_members:
                continue
            freq = M[rows].mean(axis=0)
            core = freq >= core_frac
            if core.sum() < min_group:
                continue
            # a gaussian belongs to the cluster that claims it most often
            take = core & (freq > score)
            labels[take] = g
            score[take] = freq[take]
            g += 1
        return labels, np.arange(n)

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

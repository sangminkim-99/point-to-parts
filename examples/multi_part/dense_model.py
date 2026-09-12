"""Step 4: a per-part Gaussian model grown from keyframes.

One Gaussian per depth pixel, single colour, no SH. New Gaussians are added only
where the current model's render leaves observed surface uncovered, and each one
carries the label of the part it was grown against.
"""
import numpy as np
import torch

from point2pose.pipeline.components.gaussian_part_assignment import (
    GaussianCloud, GaussianPartAssignment)


class PartGaussians:
    """Labelled Gaussians for a set of parts, in each part's anchor frame."""

    def __init__(self, rgb, depth, mask, K, part_masks, stride=2,
                 max_total=150000, device="cuda"):
        self.K, self.device, self.stride = K, device, stride
        self.max_total = max_total
        self.cloud = GaussianCloud.from_depth(rgb, depth, K, mask, stride=stride,
                                              device=device)
        if self.cloud is None:
            raise RuntimeError("anchor frame gives no gaussians")
        self.assign = GaussianPartAssignment(self.cloud, max(2, len(part_masks)),
                                             device=device)
        n = len(self.cloud)
        # a gaussian starts owned by whichever part covers its anchor pixel
        self.labels = np.full(n, -1, dtype=np.int32)
        for j, pm in enumerate(part_masks):
            if pm is not None and pm.shape[0] == n:
                self.labels[pm] = j
        self.n_initial = n
        # diagnostics only (no behaviour): the nearest sparse-seed distance that
        # drove each gaussian's last split reassignment, and the last carve's
        # per-gaussian detail. NaN = never split-assigned (grown or initial).
        self.seed_dist = np.full(n, np.nan, dtype=np.float32)
        self._carve_note = None

    # ---- membership ----
    def weights(self, j, n=None):
        n = len(self.cloud) if n is None else n
        w = np.zeros(n, np.float32)
        w[:len(self.labels)][self.labels[:n] == j] = 1.0
        return w

    def all_weights(self, n_parts):
        return [self.weights(j) for j in range(n_parts)]

    def relabel(self, idx, j):
        self.labels[idx] = j

    # ---- step 4: grow only where the render does not cover the depth ----
    def grow(self, rgb, depth, mask, poses, grow_ok=None, max_new=1500):
        if len(self.cloud) >= self.max_total:
            return 0
        ws = self.all_weights(len(poses))
        n_new, owner = self.assign.grow_parts(
            rgb, depth, self.K, mask, poses, ws, stride=self.stride,
            max_new=max_new, grow_ok=grow_ok, max_total=self.max_total)
        if not n_new:
            return 0
        self.labels = np.concatenate([self.labels, np.asarray(owner, np.int32)])
        self.seed_dist = np.concatenate(
            [self.seed_dist, np.full(n_new, np.nan, np.float32)])
        return n_new

    def uncovered(self, depth, mask, poses):
        """Fraction of observed object surface no part's model explains."""
        H, W = depth.shape
        seen = (mask > 0) & (depth > 0)
        cov = None
        for j, T in enumerate(poses):
            if T is None:
                continue
            o = self.assign.occupancy(T, self.weights(j), self.K, H, W, dilate=2)
            if o is None:
                continue
            o = o.cpu().numpy() if hasattr(o, "cpu") else o
            cov = o if cov is None else (cov | o)
        if cov is None:
            return 1.0
        return float((seen & ~cov).sum()) / max(1, int(seen.sum()))

    # ---- free-space carving, so a stale copy cannot survive ----
    def carve(self, depth, poses, margin=0.03, votes=3):
        H, W = depth.shape
        dt = torch.as_tensor(depth, dtype=torch.float32, device=self.device)
        Kt = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)
        n = len(self.cloud)
        free = np.zeros(n, bool)
        for j, T in enumerate(poses):
            if T is None:
                continue
            Tt = torch.as_tensor(T, dtype=torch.float32, device=self.device)
            q = self.cloud.means @ Tt[:3, :3].T + Tt[:3, 3]
            z = q[:, 2]
            u = (q[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
            v = (q[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
            ok = (z > 1e-3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zo = torch.zeros_like(z)
            zo[ok] = dt[v[ok].clamp(0, H - 1), u[ok].clamp(0, W - 1)]
            carve = (ok & (zo > 0) & (zo > z + margin)).cpu().numpy()
            free |= carve & (self.labels[:n] == j)
        v_ = getattr(self, "_votes", np.zeros(0, np.int16))
        if len(v_) < n:
            v_ = np.concatenate([v_, np.zeros(n - len(v_), np.int16)])
        v_[:n] = np.where(free, v_[:n] + 1, 0)
        self._votes = v_
        go = np.where(v_[:n] >= votes)[0]
        if go.size:
            # diagnostics: record what is about to be carved, and the seed
            # distance that assigned it at its last split, BEFORE overwriting
            self._carve_note = {"idx": go.copy(),
                                "labels": self.labels[go].copy(),
                                "seed_dist": self.seed_dist[go].copy()}
            self.labels[go] = -1
            v_[go] = 0
        else:
            self._carve_note = None
        return int(go.size)

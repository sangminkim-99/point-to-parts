"""Sparse-only articulated tracker: the method with nothing added.

Steps 1-3 of the plan and no more -- point tracks, a rigid fit per part, a split
when one rigid motion stops explaining the residual, and a screw axis from the
relative pose trajectory. No gaussians, no rendering, no dense energy. This is
the baseline the dense machinery has to beat, measured the same way.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from point2pose.pipeline.components.joint_model import JointModel


def lift(pts2d, depth, K):
    """Pixel + depth -> 3D, with a validity flag per point."""
    u = np.clip(np.round(pts2d[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(pts2d[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    ok = z > 0.05
    xyz = np.stack([(pts2d[:, 0] - K[0, 2]) * z / K[0, 0],
                    (pts2d[:, 1] - K[1, 2]) * z / K[1, 1], z], 1)
    return xyz.astype(np.float32), ok


@dataclass
class NaiveConfig:
    n_points: int = 100
    inlier_thres: float = 0.01
    min_inliers: int = 3
    ransac_iters: int = 200
    num_pips_iter: int = 1
    tapir_res: int = 512
    # A split is proposed when a MINORITY of points stops following the fit.
    # The median residual cannot see that: a second part is by definition the
    # smaller set, so the median stays with the majority and never rises.
    split_out_frac: float = 0.12        # share of points beyond the inlier band
    split_out_band: float = 3.0         # x inlier_thres
    split_res: float = 0.05             # metres of median residual, as a backstop
    split_frames: int = 3               # consecutive frames above either
    min_part_pts: int = 6
    ambiguous_band: float = 1.5         # x inlier_thres: points in the joint gap
    min_frames_before_split: int = 8
    resplit_wait: int = 12
    max_parts: int = 6
    joint_gate: float = 0.02
    # optional layers, each measurable on its own
    dense: bool = False                 # step 4: per-part gaussian model
    dense_stride: int = 2
    grow_every: int = 4
    refine: bool = False                # step 5: rendering-based refinement
    refine_every: int = 3
    carve_every: int = 3


@dataclass
class NaivePart:
    idx: np.ndarray                     # track indices owned by this part
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    parent: int = 0
    joint: JointModel = None
    resid: float = 0.0
    out_frac: float = 0.0
    energy: float = float("inf")
    over: int = 0                       # consecutive frames above split_res


class NaivePartTracker:
    """Point tracks in, per-part SE(3) and joints out. Nothing dense."""

    RIGID, SPLIT = "rigid", "split"

    def __init__(self, K, cfg, tracker, reg):
        self.K, self.cfg, self.tracker, self.reg = K, cfg, tracker, reg
        self.state = self.RIGID
        self.parts = []
        self.n = 0
        self.last_timings = {}
        self.split_log = []

    def start(self, rgb, depth, mask):
        from point2pose.data_types.frame import Frame
        from examples.multi_part.streaming import (build_sampler,
                                                   sample_superpoint, clean_mask)
        mask = clean_mask(mask, depth)
        self.sampler = build_sampler(self.cfg)
        pts0 = sample_superpoint(self.sampler, rgb, depth, mask, self.K,
                                 self.cfg.n_points)
        f0 = Frame(id=0, rgb=rgb, depth=depth, intrinsics=self.K)
        self.tracker.add_query_points(f0, pts0)
        self.tracker.initialize(f0)
        self.pts0 = np.asarray(pts0, np.float32)
        self.anchor_xyz, self.anchor_ok = lift(pts0, depth, self.K)
        self.parts = [NaivePart(idx=np.where(self.anchor_ok)[0])]
        self.n = 1
        self.model = self.refiner = None
        if self.cfg.dense:
            from examples.multi_part.dense_model import PartGaussians
            self.model = PartGaussians(rgb, depth, mask, self.K, [None],
                                       stride=self.cfg.dense_stride)
            self.model.labels[:] = 0        # one body to start with
            if self.cfg.refine:
                from examples.multi_part.refine import RenderRefiner
                self.refiner = RenderRefiner(self.model)
        return self

    # ---- one frame ----
    def step(self, rgb, depth, mask):
        from point2pose.data_types.frame import Frame
        cfg = self.cfg
        t0 = time.perf_counter()
        i = self.n
        self.n += 1

        tracks, _, vis = self.tracker.track_once(
            Frame(id=i, rgb=rgb, depth=depth, intrinsics=self.K))
        vis = vis.astype(bool)
        cur, cur_ok = lift(tracks, depth, self.K)
        self.cur = (cur, cur_ok, vis)
        self.cur_tracks_2d = tracks
        t_track = time.perf_counter()

        for p in self.parts:
            self._fit(p, cur, cur_ok, vis)

        if i >= cfg.min_frames_before_split and len(self.parts) < cfg.max_parts \
                and i - getattr(self, "last_split", 0) >= cfg.resplit_wait:
            for j, p in enumerate(list(self.parts)):
                if p.over >= cfg.split_frames and self._split(j, cur, cur_ok, vis):
                    self.last_split = i
                    break

        for j, p in enumerate(self.parts):
            if p.joint is None or j == p.parent or p.parent >= len(self.parts):
                continue
            if p.resid < cfg.joint_gate:
                A = np.linalg.inv(self.parts[p.parent].pose) @ p.pose
                p.joint.add(A)
                if p.joint.kind is None or len(p.joint.A) % 4 == 0:
                    p.joint.fit()

        t_dense = t_track
        if self.model is not None:
            self._dense_step(rgb, depth, mask, i)
            t_dense = time.perf_counter()

        self.last_timings = {"track_ms": (t_track - t0) * 1e3,
                             "dense_ms": (t_dense - t_track) * 1e3,
                             "total_ms": (time.perf_counter() - t0) * 1e3}
        return self

    def _dense_step(self, rgb, depth, mask, i):
        """Steps 4 and 5, run at their own rate on top of the sparse frontend."""
        from examples.multi_part.streaming import clean_mask
        cfg = self.cfg
        mask = clean_mask(mask, depth)
        H, W = depth.shape
        poses = [p.pose for p in self.parts]

        if self.refiner is not None and i % cfg.refine_every == 0:
            out = self.refiner.refine_poses(self.parts, depth, mask, self.K, H, W)
            for p, o in zip(self.parts, out):
                if o is not None:
                    p.pose, p.energy = o
            self.moved = getattr(self, "moved", 0) + self.refiner.update_labels(
                self.parts, depth, mask, self.K, H, W)
            poses = [p.pose for p in self.parts]

        if i % cfg.carve_every == 0:
            self.carved = getattr(self, "carved", 0) + \
                self.model.carve(depth, poses)
        if cfg.grow_every > 0 and i % cfg.grow_every == 0:
            ok = [p.resid < cfg.joint_gate for p in self.parts]
            self.grown = getattr(self, "grown", 0) + \
                self.model.grow(rgb, depth, mask, poses, grow_ok=ok)
        self.unmodelled = self.model.uncovered(depth, mask, poses)

    def _fit(self, part, cur, cur_ok, vis):
        """Rigid fit of this part's points from the anchor frame, RANSAC + SVD."""
        cfg = self.cfg
        sel = part.idx[self.anchor_ok[part.idx] & cur_ok[part.idx]
                       & vis[part.idx]]
        if sel.size < cfg.min_inliers or self.reg is None:
            part.over = 0
            return
        c = self.reg._RANSAC(p0=self.anchor_xyz[sel], tgt_pcd=cur[sel], w=None,
                             remaining=np.ones(sel.size, bool), init_pose=None)
        if c is None:
            part.over = 0
            return
        part.pose = np.asarray(c["T"])
        d = np.linalg.norm(
            self.anchor_xyz[sel] @ part.pose[:3, :3].T + part.pose[:3, 3]
            - cur[sel], axis=1)
        part.resid = float(np.median(d))
        part.out_frac = float(np.mean(d > cfg.split_out_band * cfg.inlier_thres))
        hot = (part.out_frac > cfg.split_out_frac) or (part.resid > cfg.split_res)
        part.over = part.over + 1 if hot else 0

    def _split(self, j, cur, cur_ok, vis):
        """Second rigid motion inside one part; ambiguous points are dropped."""
        cfg = self.cfg
        part = self.parts[j]
        sel = part.idx[self.anchor_ok[part.idx] & cur_ok[part.idx]
                       & vis[part.idx]]
        if sel.size < 2 * cfg.min_part_pts:
            return False
        rem = np.ones(sel.size, bool)
        motions = []
        for _ in range(2):
            c = self.reg._RANSAC(p0=self.anchor_xyz[sel], tgt_pcd=cur[sel],
                                 w=None, remaining=rem, init_pose=None)
            if c is None:
                break
            motions.append(np.asarray(c["T"]))
        if len(motions) < 2:
            return False
        D = np.stack([np.linalg.norm(
            self.anchor_xyz[sel] @ T[:3, :3].T + T[:3, 3] - cur[sel], axis=1)
            for T in motions])
        best = D.argmin(axis=0)
        bv, sv = D.min(axis=0), np.sort(D, axis=0)[1]
        # a point in the joint gap fits both motions; it belongs to neither
        keep = (bv < cfg.inlier_thres) & (sv > cfg.ambiguous_band * bv)
        groups = [sel[keep & (best == k)] for k in range(2)]
        if min(len(g) for g in groups) < cfg.min_part_pts:
            return False
        order = np.argsort([-len(g) for g in groups])
        new = [NaivePart(idx=groups[k], pose=motions[k]) for k in order]
        self.parts[j:j + 1] = new
        for k, p in enumerate(self.parts):
            p.parent = 0 if k else (1 if len(self.parts) > 1 else 0)
            if p.joint is None and k != p.parent:
                p.joint = JointModel()
        if self.model is not None:
            self._split_labels(j, new, cur, cur_ok)
        self.state = self.SPLIT
        self.split_log.append((self.n, len(self.parts)))
        print(f"[naive] split at frame {self.n}: {len(self.parts)} parts "
              f"(residual {part.resid * 1000:.0f} mm, "
              f"outliers {100 * part.out_frac:.0f}%, "
              f"{len(groups[0])}/{len(groups[1])} points, "
              f"{int((~keep).sum())} dropped)")
        return True

    def _split_labels(self, j, new, cur, cur_ok):
        """Give the split part's gaussians to whichever half is nearer in 3D."""
        from scipy.spatial import cKDTree
        gm = self.model.cloud.means.detach().cpu().numpy()
        held = np.where(self.model.labels[:len(gm)] == j)[0]
        if held.size == 0:
            return
        # every part after j shifts up by one, so renumber from the back
        for k in range(len(self.parts) - 1, j, -1):
            self.model.labels[self.model.labels == k - 1] = k
        seeds, owner = [], []
        for k, p in enumerate(new):
            q = self.anchor_xyz[p.idx[self.anchor_ok[p.idx]]]
            if len(q):
                seeds.append(q)
                owner.append(np.full(len(q), j + k))
        if len(seeds) < 2:
            return
        pts = np.concatenate(seeds)
        own = np.concatenate(owner)
        _, nn = cKDTree(pts).query(gm[held], k=1)
        self.model.labels[held] = own[nn]

    # ---- visualisation ----
    def render(self, bgr, palette=None):
        import cv2
        pal = palette or [(60, 140, 235), (200, 120, 40), (70, 180, 90),
                          (200, 80, 200), (60, 200, 200)]
        cur, cur_ok, vis = getattr(self, "cur", (None, None, None))
        t2 = getattr(self, "cur_tracks_2d", None)
        if t2 is None:
            return bgr
        owner = np.full(len(t2), -1)
        for j, p in enumerate(self.parts):
            owner[p.idx[p.idx < len(t2)]] = j
        for k, (x, y) in enumerate(t2):
            if not (cur_ok[k] and vis[k]):
                continue
            col = (110, 110, 110) if owner[k] < 0 else pal[owner[k] % len(pal)]
            cv2.circle(bgr, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)
        for j, p in enumerate(self.parts):
            q = t2[p.idx[p.idx < len(t2)]]
            if len(q) < 3:
                continue
            x0, y0 = q.min(0).astype(int)
            x1, y1 = q.max(0).astype(int)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), pal[j % len(pal)], 2)
            lab = f"p{j} {p.resid * 1000:.0f}mm {100 * p.out_frac:.0f}%"
            if p.joint is not None and p.joint.kind:
                lab += f" {p.joint.kind[:4]}"
            cv2.putText(bgr, lab, (x0, max(12, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        pal[j % len(pal)], 1, cv2.LINE_AA)
        return bgr

"""Online part discovery and per-part 6-DoF tracking, one frame at a time.

The offline experiment in `experiments/articulated/run_gaussian_part_discovery.py`
is two-pass: it walks the whole sequence accumulating evidence, groups once at the
end, then walks it again to track each part. That is fine for a benchmark and
useless for a camera. This is the same method restructured as a state machine that
never looks forward:

    RIGID   one body, one box. Motion hypotheses are proposed and assigned every
            frame, and the evidence (co-association, winner sets, log-odds) piles
            up. Every `regroup_every` frames the accumulated evidence is offered
            to the same ground-truth-free model selection the benchmark uses. If
            it prefers a split over a single body, the split is committed.

    SPLIT   each part carries its own membership, dense model, 6-DoF pose and
            joint. Tracking is forward-only: the previous pose, the current
            hypotheses, a screw extrapolation and a sweep of the fitted joint are
            all refined and the best-explaining one wins.

Timing on an RTX-class GPU, measured: the dense stage costs about 22 ms/frame and
is nearly independent of cloud size, because it is dominated by the kernel launches
of the ICP loop rather than by the number of gaussians. Point tracking and RANSAC
sit on top of that.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from point2pose.pipeline.components.gaussian_part_assignment import (
    GaussianCloud,
    GaussianPartAssignment,
)
from point2pose.pipeline.components.joint_model import JointModel
from scipy.spatial.transform import Rotation as _R


def se3_pow(T, s):
    """T raised to a real power, along its own screw axis."""
    from scipy.linalg import expm, logm

    if abs(s - 1.0) < 1e-9:
        return np.asarray(T)
    try:
        return np.real(expm(np.real(logm(np.asarray(T, dtype=np.float64))) * s))
    except Exception:
        return np.asarray(T)


SAMPLERS = ("super_point_balanced", "super_point_fps", "super_point",
            "uniform_fps", "orb", "random")


def build_sampler(cfg):
    """Point2Pose's own sampler, so the few points we get are worth tracking.

    pipeline_test2.yaml runs `super_point_balanced` at 20 points; 20 RANDOM
    points is a different thing entirely -- measured, it never separates the
    eyeglasses and shrinks the storage parts to a few hundred gaussians.
    """
    kind = getattr(cfg, "sampler", "super_point_balanced")
    # Coverage matters more here than raw keypoint score: a part is found from
    # points that sit on it, so a cluster on one textured corner is worse than
    # a thinner spread over the whole object.
    common = {
        "num_points": cfg.n_points, "density_per_kpx": -1,
        "min_points": 5, "max_points": max(cfg.n_points, 50),
        "edge_margin_px": 5, "remove_convex_hull": False,
        "inflate_points": False, "crop_to_mask": True, "crop_pad_px": 3,
        "cell_size": getattr(cfg, "sampler_cell", -1),
        "super_point_max_num_keypoints": 1024, "debug_level": 0,
    }
    if kind == "super_point_balanced":
        common.update({
            "fps_oversample_factor": 4,
            "nms_radius_px": getattr(cfg, "sampler_nms", 0.0),
            "score_weight": getattr(cfg, "sampler_score_w", 0.20),
            "min_separation_px": getattr(cfg, "sampler_min_sep", 6.0),
            "separation_penalty_weight": 0.5,
        })
    elif kind in ("super_point_fps", "super_point"):
        common.update({"fps_oversample_factor": 4,
                       "cell_size": max(1, getattr(cfg, "sampler_cell", 8))})
    elif kind == "uniform_fps":
        # geometric spread with no texture requirement -- the interesting
        # counterpoint to SuperPoint, and the one likeliest to be untrackable
        common.update({"density_per_kpx": -1})
    try:
        from point2pose.core.build import build_from_cfg
        from point2pose.core.module_registry import SAMPLER
        import point2pose.modules.sampler  # noqa: F401  (registers them)
        return build_from_cfg({"type": kind, "params": common}, SAMPLER)
    except Exception as exc:
        print(f"[stream] sampler '{kind}' unavailable ({exc}); using random")
        return None


def sample_superpoint(sampler, rgb, depth, mask, K, n, min_px=20000):
    """Run the SuperPoint sampler over one mask; falls back to random.

    A small object yields almost no keypoints -- RBO pliers cover 1.5% of the
    frame and give 12 -- so the crop is upscaled until it is worth detecting on.
    """
    import torch as _t
    from point2pose.data_types.frame import Frame
    from point2pose.data_types.sampler_context import SamplerContext

    if sampler is None:
        return sample_points(mask, depth, n)
    import cv2
    area = int((mask > 0).sum())
    up = 1
    if 0 < area < min_px:
        up = int(min(4, max(1, round(np.sqrt(min_px / max(area, 1))))))
    if up > 1:
        rgb = cv2.resize(rgb, None, fx=up, fy=up, interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, None, fx=up, fy=up,
                           interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask.astype(np.uint8), None, fx=up, fy=up,
                          interpolation=cv2.INTER_NEAREST)
        K = np.asarray(K).copy(); K[:2] *= up
    m = _t.as_tensor((mask > 0).astype(np.uint8))[None, None]
    f = Frame(id=0, rgb=rgb, depth=depth, mask=m, intrinsics=K)
    try:
        pts = sampler.sample(SamplerContext(frame=f, min_depth=0.1, max_depth=10.0), 0)
    except Exception as exc:
        print(f"[stream] SuperPoint sample failed ({exc}); using random")
        return sample_points(mask, depth, n)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2) / up
    if pts.shape[0] < 5:
        return sample_points(mask, depth, n).astype(np.float32) / up
    return pts


def clean_mask(mask, depth, jump=0.15, win=7):
    """Drop mask pixels that sit well BEHIND the object's local surface.

    A propagated mask leaks background at the silhouette, and the leak is always
    behind. A symmetric depth-span test also deletes the object's own edge,
    which on a live sensor eats enough of the mask that the poses stop updating
    and the model is drawn at its old place over a moving image.
    """
    import cv2
    m = (mask > 0) & (depth > 0)
    if jump <= 0 or not m.any():
        return m.astype(np.uint8) * 255
    # nearest masked surface within `win`: erosion of depth is a min filter, and
    # unmasked pixels are pushed to +inf so they cannot win it
    big = np.where(m, depth, np.float32(1e3)).astype(np.float32)
    near = cv2.erode(big, np.ones((win, win), np.uint8))
    out = m & (depth <= near + jump)
    if out.sum() < 0.5 * m.sum():       # the object really is that deep
        return m.astype(np.uint8) * 255
    n, lb, st, _ = cv2.connectedComponentsWithStats(out.astype(np.uint8), 8)
    if n > 2:                            # keep the body, drop specks
        keep = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        out = lb == keep
    return out.astype(np.uint8) * 255


def sample_points(mask, depth, n, border=4):
    """Pick n pixels inside the mask that have valid depth."""
    m = (mask > 0) & (depth > 0.05)
    m[:border] = m[-border:] = m[:, :border] = m[:, -border:] = False
    ys, xs = np.where(m)
    if len(ys) == 0:
        return np.zeros((0, 2), np.float32)
    idx = np.random.default_rng(0).choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def lift(pts, depth, K):
    """Pixels -> camera-frame 3D, with a validity flag."""
    u = np.clip(pts[:, 0].astype(int), 0, depth.shape[1] - 1)
    v = np.clip(pts[:, 1].astype(int), 0, depth.shape[0] - 1)
    z = depth[v, u]
    ok = z > 0.05
    xyz = np.stack([(u - K[0, 2]) * z / K[0, 0],
                    (v - K[1, 2]) * z / K[1, 1], z], 1).astype(np.float32)
    return xyz, ok


@dataclass
class Part:
    """One discovered part: what it is made of, where it is, and how it hinges."""
    weights: np.ndarray                 # soft membership over gaussians
    pose: np.ndarray = field(default_factory=lambda: np.eye(4))
    energy: float = float("inf")
    best_energy: float = float("inf")   # best this part has ever explained at
    box: object = None                  # oriented box in the anchor frame
    joint: JointModel = None
    parent: int = 0
    joint_values: list = field(default_factory=list)
    d_pose: np.ndarray = None           # last inter-frame increment
    track_idx: np.ndarray = None        # sparse tracks sitting on this part
    view_dirs: list = None              # viewing directions already keyframed
    opt: object = None                  # per-part ISAM2 graph
    opt_n: int = 0


@dataclass
class Config:
    max_hyp: int = 4
    gauss_target: int = 3000
    gauss_stride_max: int = 2
    n_points: int = 20
    inlier_thres: float = 0.01
    min_inliers: int = 3
    depth_sigma: float = 0.02
    win_steps: int = 20
    ransac_iters: int = 50
    # 4 costs 260 ms at 224 points, 1 costs 45 ms -- but 1 also costs quality:
    # tracking energy ~10x worse, and pliers' joint comes out prismatic not revolute.
    num_pips_iter: int = 1
    # Not free to raise: after the split each part still fits its own RANSAC
    # from its own tracks, which a thin fast-rotating part depends on.
    hyp_every: int = 1
    win_min_span: int = 4
    # Clustering this is ~cubic: a split attempt costs 660-860 ms at 4000 and
    # 46-52 ms at 1000, for the same parts on pliers, storage and eyeglasses.
    co_sample: int = 1000
    min_group: int = 30
    # An absolute floor is the wrong unit: 30 is 3% of the 1000-gaussian
    # co-association subsample but 0.1% of a 30k cloud, so the final filter let
    # 54-gaussian slivers through as parts.
    min_group_frac: float = 0.02
    # which grouping candidates to generate; drop one to ablate it
    groupings: str = "labels,coassoc,winsets,merge"
    # groupings: str = "labels"
    # (25, 10) -> (12, 3): split at frame 25 not 31, energy 0.02 -> 0.003,
    # because splitting earlier leaves more frames for each model to grow.
    min_frames_before_split: int = 12
    # part discovery was one-shot: once SPLIT, _try_split was never called
    # again, so a third part could not be found no matter what moved
    # Membership was frozen at the split, so a part kept a copy of its
    # neighbour's surface from frame 0 and drew it at its old place forever.
    # Co-Fusion (Runz & Agapito 2017) removes such geometry from the model it
    # was wrongly left in rather than moving it, and EM-Fusion (Strecke &
    # Stueckler 2019) re-infers the association every frame with an explicit
    # "belongs to nothing" outlier term.
    # EM-Fusion's uniform outlier class: without it a gaussian that fits no part
    # still has to pick one. Metres; 0 disables.
    outlier_r: float = 0.0
    # Point2Pose's own back end. Each part's pose was chosen greedily per frame
    # with nothing to pull it back once it slipped: RBO median 57 mm but p90
    # 544 mm. A pose graph over the part's own tracks is what Point2Pose already
    # brings and what the streaming tracker was not using at all.
    graph: bool = False   # measured on RBO: helps one part, wrecks another
    graph_relin: float = 0.1
    graph_prior: float = 0.05
    prune: bool = True
    prune_votes: int = 3
    prune_every: int = 3
    prune_margin: float = 0.03          # metres the sensor must see past the model
    resplit: bool = True
    resplit_wait: int = 12              # frames of evidence after a split
    # A re-split has to earn its extra part. Without a floor the grouping with
    # the required group count wins even at ratio 0.00, and every object
    # fragments to max_hyp pieces.
    # The grouping score rises monotonically with the number of groups -- every
    # extra group picks its own best hypothesis -- so comparing a re-split
    # against a fixed constant can never say no. RBO cabinet01 walked 2 -> 6
    # parts with the ratio climbing 1.16, 1.20, 1.27, 1.36 at every step. The
    # comparison has to be against the partition already in use.
    resplit_min_ratio: float = 1.02     # kept for the absolute floor
    resplit_gain: float = 1.08          # x the CURRENT partition's own score
    # Relative to the coverage the accepted split itself reached. An absolute
    # 0.30 was calibrated on sim (first split 0.82-0.94); on a real sequence
    # 87% of gaussians are undecided, so coverage cannot exceed ~0.13 and the
    # floor was unreachable.
    resplit_cov_frac: float = 0.5
    regroup_every: int = 3
    pose_log_max: int = 120
    # per-part tracking
    track_rmax: float = 0.05
    # extra penalty for rendering outside the live mask, as a multiple of r_max
    outside_w: float = 0.0  # measured: helps 2 objects, badly hurts 3. off.
    # pipeline_test2.yaml pose_jump_guard_trans_thres
    max_step: float = 0.15
    grow_every: int = 2
    # Relative to the part's own best, not absolute. An absolute 0.006 (copied
    # from map_growth_max_mean_residual, which is a different quantity) blocked
    # 17 of 23 growth attempts because part energies sit at 0.02-0.05.
    grow_gate_rel: float = 1.6
    grow_gate_abs: float = 0.05
    # a pose slightly off the surface is still informative about the joint, and
    # the fit re-evaluates both joint types anyway
    joint_gate: float = 0.035
    grow_max: int = 1500
    # a live stream has no end, so the model needs a ceiling
    gauss_max: int = 150000
    winset_max: int = 400
    # Point2Pose's live sampling criterion, per part (pipeline_test2.yaml:
    # rotation_threshold max_angle_deg 15, sampler num_points 20 / max 50).
    key_angle_deg: float = 15.0
    key_min_pts: int = 8
    key_points: int = 20
    key_max_points: int = 400
    exclusive_mask: bool = True
    # Off: measured, masking a part's back face out costs more than the
    # thickness it saves -- laptop lid 23.6 -> 33.4 mm, eyeglasses temple 38 -> 168.
    visibility: bool = False
    score_round: int = 3
    merge_eps: float = 0.002
    merge_tol: float = 0.012
    # Metres a mask pixel may sit behind the nearest surface in its mask_win
    # neighbourhood before it counts as background leaking through the
    # silhouette. A 21 px reference also rejects a thin part that is genuinely
    # behind the body (sim eyeglasses 3 -> 2 parts); 7 px keeps both.
    mask_depth_jump: float = 0.15
    mask_win: int = 7
    # spatial spread of the sampled points; coverage beats keypoint score here
    # Measured on RBO: NMS and one-per-cell rejection buy 4 points of coverage
    # on a big object and cost a third of the points on a small one. Only the
    # novelty weight is worth changing.
    sampler: str = "super_point_balanced"
    sampler_cell: int = -1
    sampler_nms: float = 0.0
    sampler_score_w: float = 0.20
    sampler_min_sep: float = 6.0


class StreamingPartDiscovery:
    """Discover parts and track their 6-DoF poses from a live RGB-D stream."""

    RIGID = "rigid"
    SPLIT = "split"

    def __init__(self, K, cfg=None, tracker=None, register=None):
        self.K = np.asarray(K, dtype=np.float64)
        self.cfg = cfg or Config()
        self.tracker = tracker
        self.reg = register
        self.state = self.RIGID
        self.n = 0
        self.parts = []
        self.cloud = None
        self.assign = None
        self.last_timings = {}
        self.diag = {}
        self.debug = True

    # ---- anchor frame ----
    def start(self, rgb, depth, mask):
        """Lift the first masked frame to gaussians and seed the point tracks."""
        from point2pose.data_types.frame import Frame

        cfg = self.cfg
        H, W = depth.shape
        self.H, self.W = H, W
        mask = clean_mask(mask, depth, cfg.mask_depth_jump, cfg.mask_win)

        # a thin object at a fixed stride yields a few hundred gaussians, and
        # then nothing is ever decisively assigned
        gs = cfg.gauss_stride_max
        while gs > 1 and int((mask[::gs, ::gs] > 0).sum()) < cfg.gauss_target:
            gs -= 1
        self.gauss_stride = gs

        self.cloud = GaussianCloud.from_depth(rgb, depth, self.K, mask, stride=gs)
        if self.cloud is None or len(self.cloud) < 50:
            raise RuntimeError("anchor frame gives too few gaussians")
        self.assign = GaussianPartAssignment(
            self.cloud, cfg.max_hyp, depth_sigma=cfg.depth_sigma)
        self.assign.outlier_r = cfg.outlier_r
        self.assign.init_coassoc(min(cfg.co_sample, len(self.cloud)))

        self.sampler = build_sampler(cfg)
        pts0 = sample_superpoint(self.sampler, rgb, depth, mask, self.K,
                                 cfg.n_points)
        f0 = Frame(id=0, rgb=rgb, depth=depth, intrinsics=self.K)
        self.tracker.add_query_points(f0, pts0)
        self.tracker.initialize(f0)
        self.anchor_xyz, self.anchor_ok = lift(pts0, depth, self.K)

        self.past = {}          # frame index -> (xyz, ok, visible)
        self.order = []
        self.anchor_to = {}
        self.pose_log = []
        self.n = 1
        self.whole_pose = np.eye(4)
        self.n_initial = len(self.cloud)
        self.whole_box = self._fit_box(np.ones(len(self.cloud), bool))
        return self

    # ---- one frame ----
    def step(self, rgb, depth, mask):
        from point2pose.data_types.frame import Frame

        t0 = time.perf_counter()
        i = self.n
        self.n += 1
        cfg = self.cfg
        mask = clean_mask(mask, depth, cfg.mask_depth_jump, cfg.mask_win)

        tracks, _, vis = self.tracker.track_once(
            Frame(id=i, rgb=rgb, depth=depth, intrinsics=self.K))
        vis = vis.astype(bool)
        cur, cur_ok = lift(tracks, depth, self.K)
        self.past[i] = (cur.copy(), cur_ok.copy(), vis.copy())
        self.order.append(i)
        # keep only what the trailing window can still reach
        for j in list(self.past):
            if i - j > cfg.win_steps + 2:
                self.past.pop(j, None)
        t_track = time.perf_counter()

        self.cur_tracks = (cur, cur_ok, vis)
        self.cur_tracks_2d = tracks
        if cfg.hyp_every <= 1 or i % cfg.hyp_every == 0 \
                or not getattr(self, "_last_hyps", None):
            hyps = self._hypotheses(i, cur, cur_ok, vis)
            self._last_hyps = hyps
        else:
            hyps = self._last_hyps
            self.anchor_to[i] = self.anchor_to.get(self.order[-2], np.eye(4))
        t_hyp = time.perf_counter()

        if self.state == self.RIGID:
            self._step_rigid(rgb, depth, mask, hyps, i)
        else:
            self._step_split(rgb, depth, mask, hyps, i)
        t_end = time.perf_counter()

        n_uni = len({tuple(np.round(np.asarray(T).ravel(), 4)) for T in hyps})
        _, cok, cvi = self.cur_tracks
        self.diag = {
            "hyp": n_uni,
            "tracks_live": int((cok & cvi).sum()),
            "tracks": int(len(cok)),
            "decisive": float(getattr(self.assign, "last_decisive", 0.0)),
            "tries": getattr(self, "split_tries", 0),
            "gauss": len(self.cloud),
            "winsets": len(getattr(self.assign, "winsets", [])),
            "why": getattr(self, "last_split_why", "no attempt yet"),
        }
        self.last_timings = {
            "track_ms": (t_track - t0) * 1e3,
            "hyp_ms": (t_hyp - t_track) * 1e3,
            "dense_ms": (t_end - t_hyp) * 1e3,
            "total_ms": (t_end - t0) * 1e3,
        }
        return self.state

    def _hypotheses(self, i, cur, cur_ok, vis):
        """Rigid-motion candidates, all expressed anchor -> current."""
        cfg = self.cfg
        # the anchor-to-current dominant motion, kept every frame so a windowed
        # fit can always be composed back into the frame the gaussians live in
        u0 = np.where(self.anchor_ok & cur_ok & vis)[0]
        Ta = None
        if len(u0) >= cfg.min_inliers and self.reg is not None:
            c = self.reg._RANSAC(p0=self.anchor_xyz[u0], tgt_pcd=cur[u0], w=None,
                                 remaining=np.ones(len(u0), bool), init_pose=None)
            if c is not None:
                Ta = c["T"]
        if Ta is None:
            Ta = self.anchor_to.get(self.order[-2], np.eye(4)) \
                if len(self.order) > 1 else np.eye(4)
        self.anchor_to[i] = Ta
        for j in list(self.anchor_to):
            if i - j > cfg.win_steps + 2:
                self.anchor_to.pop(j, None)

        hyps = []
        for span in (cfg.win_steps, cfg.win_steps // 2, cfg.win_steps // 4):
            if span < cfg.win_min_span or len(self.order) <= span:
                continue
            j = self.order[-1 - span]
            if j not in self.past or j not in self.anchor_to:
                continue
            sp, so, sv = self.past[j]
            u = np.where(so & sv & cur_ok & vis)[0]
            # if len(u) < 2 * cfg.min_inliers:
            if len(u) < cfg.min_inliers:
                continue
            rem = np.ones(len(u), bool)
            got = []
            for _ in range(cfg.max_hyp):
                c = self.reg._RANSAC(p0=sp[u], tgt_pcd=cur[u], w=None,
                                     remaining=rem, init_pose=None)
                if c is None:
                    break
                got.append(c["T"] @ self.anchor_to[j])
            if got:
                hyps.extend(got)
                break
        hyps.append(Ta)

        uniq = []
        for T in hyps:
            if all(np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.008 or
                   np.linalg.norm(T[:3, :3] - U[:3, :3]) > 0.02 for U in uniq):
                uniq.append(T)
        hyps = uniq[:cfg.max_hyp]
        while len(hyps) < cfg.max_hyp:
            hyps.append(hyps[-1])
        return hyps

    # ---- RIGID: accumulate evidence, try to split from time to time ----
    def _step_rigid(self, rgb, depth, mask, hyps, i):
        cfg = self.cfg
        a = self.assign
        H, W = self.H, self.W

        hy = list(hyps)
        R = a.residuals(hy, self.K, H, W, depth, obs_mask=mask)
        P = a.soft_membership(R)
        hy = [a.refine_pose(T, P[k], self.K, H, W, depth, obs_mask=mask)
              for k, T in enumerate(hy)]
        a.step_pointwise(hy, self.K, H, W, depth, obs_mask=mask)

        Pm = a.soft_membership(
            a.residuals(hy, self.K, H, W, depth, obs_mask=mask))
        self.pose_log.append({"frame": i, "T": [np.asarray(T) for T in hy],
                              "P": Pm.detach().cpu().numpy()})
        if len(self.pose_log) > cfg.pose_log_max:
            self.pose_log.pop(0)

        # the whole object still reads as one body: show the dominant motion
        self.whole_pose = np.asarray(hy[int(Pm.sum(dim=1).argmax())])
        self.last_hyp_set = list(hy)
        self.last_share = (Pm.sum(dim=1) / max(float(Pm.sum()), 1e-6)).cpu().numpy()

        if (i >= cfg.min_frames_before_split
                and i % cfg.regroup_every == 0):
            t0 = time.perf_counter()
            self._try_split()
            self.last_split_ms = (time.perf_counter() - t0) * 1e3

    def _try_split(self, min_groups=2, min_ratio=0.0, min_cov=0.0,
                   max_groups=None, require_relative_motion=False):
        """Offer the accumulated evidence to the model selection. Commit only a
        grouping that beats treating the object as one body."""
        cfg = self.cfg
        a = self.assign
        n = len(self.cloud)

        on = {t.strip() for t in cfg.groupings.split(",") if t.strip()}
        cand = []
        if "labels" in on:
            cand.append(("labels", a.labels()))
        try:
            if "coassoc" not in on:
                raise RuntimeError("disabled")
            lab_co, sub_co = a.coassoc_labels(0.5, cfg.min_group, max_k=5)
            full = np.full(n, -1, dtype=int)
            full[sub_co] = lab_co
            cand.append(("coassoc", full))
        except Exception:
            pass
        try:
            if "winsets" not in on:
                raise RuntimeError("disabled")
            lab_ws, _ = a.winset_labels(min_members=3, thresh=0.5)
            if (lab_ws >= 0).any():
                cand.append(("winsets", lab_ws))
        except Exception:
            pass

        merged = []
        for nm, lb in (cand if "merge" in on else []):
            mf = self._merge_rigid(lb)
            if (mf >= 0).any() and \
                    len(set(mf[mf >= 0].tolist())) < len(set(lb[lb >= 0].tolist())):
                merged.append((nm + "+merge", mf))
        cand.extend(merged)

        if not cand:
            self.last_split_why = f"no grouping candidates enabled ({cfg.groupings})"
            return False
        scored = [(nm, lb) + self._grouping_score(lb) for nm, lb in cand]
        scored.sort(key=lambda x: (-round(x[2], cfg.score_round), -x[3],
                                   len(set(x[1][x[1] >= 0].tolist()))))
        # prefer the merged variant of the winner within the measured window
        best = scored[0]
        if not best[0].endswith("+merge"):
            for e in scored[1:]:
                if e[0] == best[0] + "+merge" and \
                        round(e[2], cfg.score_round) >= round(best[2], cfg.score_round) - cfg.merge_eps:
                    best = e
                    break

        # A floor without a ceiling let a 5-group candidate take the model from
        # 2 parts to 6 in one step; discovery should add one part at a time.
        def kept(lb):
            fl = max(cfg.min_group,
                     int(cfg.min_group_frac * max(1, int((lb >= 0).sum()))))
            return [g for g in sorted(set(lb[lb >= 0].tolist()))
                    if (lb == g).sum() >= fl]

        if max_groups is not None and len(kept(best[1])) > max_groups:
            for e in scored:
                if min_groups <= len(kept(e[1])) <= max_groups:
                    best = e
                    break

        self.last_scored = [(nm, len(set(lb[lb >= 0].tolist())), r, c)
                            for nm, lb, r, c in scored]
        name, lab, ratio, cov = best
        groups = sorted(g for g in set(lab[lab >= 0].tolist()) if g >= 0)
        floor = max(cfg.min_group,
                    int(cfg.min_group_frac * max(1, int((lab >= 0).sum()))))
        groups = [g for g in groups if (lab == g).sum() >= floor]
        self.split_tries = getattr(self, "split_tries", 0) + 1
        if (len(groups) < min_groups or ratio < min_ratio or cov < min_cov
                or (max_groups is not None and len(groups) > max_groups)):
            self.last_split_why = (
                f"best '{name}' ratio {ratio:.3f} cov {cov:.2f} -> "
                f"{len(groups)} group(s) >= {floor} gaussians "
                f"(need {min_groups} groups, ratio {min_ratio:.2f}, cov {min_cov:.2f})")
            if self.debug:
                print(f"[split {self.split_tries}] " + "  ".join(
                    f"{nm}:r={r:.3f},c={c:.2f}[{ng}]"
                    for nm, ng, r, c in self.last_scored))
                print("             " + self._why_no_split())
            return False

        # An extra part has to actually move relative to its parent. _merge_rigid
        # already measures that; as one candidate among many it never gated
        # anything, so a re-split kept adding parts that move together.
        if require_relative_motion:
            mf = self._merge_rigid(lab)
            if len(set(mf[mf >= 0].tolist())) < len(groups):
                self.last_split_why = (
                    f"best '{name}' splits into {len(groups)} but the groups are "
                    f"rigid to each other")
                if self.debug:
                    print(f"[split {self.split_tries}] {self.last_split_why}")
                return False

        prev = list(self.parts)
        gm = self.cloud.means.detach().cpu().numpy()
        # A re-split labels only the decisive gaussians -- 8% of the cloud on a
        # real sequence. Rebuilding the parts from that alone throws away the
        # 92% that were already correctly partitioned, which is what scrambles
        # parts that had been clean. Carry the undecided ones over instead.
        if prev:
            lab = self._carry_over(lab, groups, prev, gm)
            groups = [g for g in groups if (lab == g).any()]
        self.parts = []
        for j, g in enumerate(groups):
            w = (lab == g).astype(np.float32)
            part = Part(weights=w, box=self._fit_box(w > 0.5),
                        track_idx=self._tracks_on(gm, w))
            # a re-split must not throw away the pose the old part had: the new
            # group inherits from whichever part holds most of its gaussians
            if prev:
                ov = [float((w * q.weights[:len(w)]).sum()) for q in prev]
                src = prev[int(np.argmax(ov))]
                if max(ov) > 0:
                    part.pose = src.pose.copy()
                    part.d_pose = None if src.d_pose is None else src.d_pose.copy()
                    part.view_dirs = None if src.view_dirs is None else list(src.view_dirs)
            self.parts.append(part)
        # each part is jointed to the biggest OTHER part
        order = np.argsort([-p.weights.sum() for p in self.parts])
        self.parts = [self.parts[k] for k in order]
        for j, p in enumerate(self.parts):
            p.parent = 1 if j == 0 else 0
            p.joint = JointModel() if len(self.parts) > 1 else None
        was = self.state
        self.state = self.SPLIT
        self.last_split_frame = self.n
        self.split_info = {"grouping": name, "ratio": ratio, "coverage": cov,
                           "frame": self.n, "n_parts": len(self.parts)}
        print(f"[stream] {'re-split' if was == self.SPLIT else 'split'} at frame "
              f"{self.n}: {len(self.parts)} parts from '{name}' "
              f"(ratio {ratio:.3f}, coverage {cov:.2f})")
        return True

    def _why_no_split(self):
        """Separate the three ways a split fails: one motion, one-sided
        assignment, or evidence that never became decisive."""
        hy = getattr(self, "last_hyp_set", None) or []
        uniq, sep = [], 0.0
        for T in hy:
            T = np.asarray(T)
            if all(np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.008 or
                   np.linalg.norm(T[:3, :3] - U[:3, :3]) > 0.02 for U in uniq):
                uniq.append(T)
        for a_ in uniq:
            for b_ in uniq:
                D = np.linalg.inv(a_) @ b_
                ang = np.degrees(np.arccos(np.clip(
                    (np.trace(D[:3, :3]) - 1) / 2, -1, 1)))
                sep = max(sep, ang + 100 * np.linalg.norm(D[:3, 3]))
        share = getattr(self, "last_share", np.zeros(1))
        lo = self.assign.logodds
        top2 = lo.topk(2, dim=1).values
        margin = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        lab = self.assign.labels()
        sizes = {int(g): int((lab == g).sum()) for g in set(lab.tolist()) if g >= 0}
        return (f"hyp {len(uniq)} distinct / {len(hy)} (sep {sep:.1f}) "
                f"share {np.round(share, 2).tolist()} | "
                f"margin med {np.median(margin):.2f} p90 {np.percentile(margin, 90):.2f} "
                f"(need 1.0) | undecided {int((lab < 0).sum())}/{len(lab)} "
                f"| groups {sizes}")

    def _carry_over(self, lab, groups, prev, gm):
        """Give every gaussian the new grouping left undecided back to a part:
        the new group nearest it that came from the same old part."""
        lab = lab.copy()
        n = min(len(lab), gm.shape[0])
        lab = lab[:n]
        # which old part each new group came from
        src = {}
        for g in groups:
            m = lab == g
            ov = [float((m & (q.weights[:n] > 0.5)).sum()) for q in prev]
            src[g] = int(np.argmax(ov)) if ov and max(ov) > 0 else -1
        free = np.where(lab < 0)[0]
        if free.size == 0:
            return lab
        old = np.full(n, -1, dtype=int)
        for q_i, q in enumerate(prev):
            old[(q.weights[:n] > 0.5)] = q_i
        try:
            from scipy.spatial import cKDTree
        except Exception:
            return lab
        for q_i in set(old[free].tolist()):
            heirs = [g for g in groups if src[g] == q_i]
            if q_i < 0 or not heirs:
                continue
            take = free[old[free] == q_i]
            pool = np.where(np.isin(lab, heirs))[0]
            if pool.size == 0:
                continue
            if len(heirs) == 1:
                lab[take] = heirs[0]
                continue
            _, j = cKDTree(gm[pool]).query(gm[take], k=1)
            lab[take] = lab[pool[j]]
        return lab

    def _grouping_score(self, labels_full):
        """How much better does splitting explain the recent posteriors than not
        splitting, and how much of the object does the grouping label."""
        gs = [g for g in set(labels_full.tolist()) if g >= 0]
        if not self.pose_log or not gs:
            return 0.0, 0.0
        tot, cnt = 0.0, 0
        for rec in self.pose_log[::3]:
            P = rec["P"]
            # The cloud grows during SPLIT, so demanding an exact width skipped
            # EVERY record and returned ratio 0.000 for every candidate --
            # no re-split could ever clear its floor. Gaussians are appended,
            # never reordered, so the common prefix is aligned.
            n = min(P.shape[1], labels_full.shape[0])
            if n < 20:
                continue
            lf = labels_full[:n]
            union = lf >= 0
            if union.sum() < 20:
                continue
            base = float(P[:, :n][:, union].sum(axis=1).max())
            split = 0.0
            for g in gs:
                m = lf == g
                if m.sum() < 10:
                    continue
                split += float(P[:, :n][:, m].sum(axis=1).max())
            if base > 1e-9:
                tot += split / base
                cnt += 1
        return tot / max(cnt, 1), float((labels_full >= 0).mean())

    def _merge_rigid(self, labels_full):
        """Merge groups that never move relative to each other."""
        cfg = self.cfg
        gs = sorted(g for g in set(labels_full.tolist()) if g >= 0)
        if len(gs) < 2 or not self.pose_log:
            return labels_full
        gm = self.cloud.means.detach().cpu().numpy()
        pts, seq = {}, {g: [] for g in gs}
        for g in gs:
            m = np.where(labels_full == g)[0]
            if m.size > 400:
                m = m[np.linspace(0, m.size - 1, 400).astype(int)]
            pts[g] = gm[m]
        for rec in self.pose_log[::2]:
            P, Ts = rec["P"], rec["T"]
            n = min(P.shape[1], labels_full.shape[0])
            if n < 20:
                continue
            for g in gs:
                m = (labels_full[:n] == g)
                seq[g].append(
                    np.asarray(Ts[int(np.argmax(P[:, :n][:, m].sum(axis=1)))])
                    if m.sum() >= 10 else None)
        parent = {g: g for g in gs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for ii, A in enumerate(gs):
            for B in gs[ii + 1:]:
                X = np.concatenate([pts[A], pts[B]], 0)
                d = [float(np.linalg.norm(
                        (X @ Ta[:3, :3].T + Ta[:3, 3])
                        - (X @ Tb[:3, :3].T + Tb[:3, 3]), axis=1).mean())
                     for Ta, Tb in zip(seq[A], seq[B]) if Ta is not None and Tb is not None]
                # a high percentile, not the median: two groups are one body only
                # if they NEVER move apart
                if d and float(np.percentile(d, 90)) < cfg.merge_tol:
                    ra, rb = find(A), find(B)
                    if ra != rb:
                        parent[rb] = ra
        remap, nxt = {}, 0
        out = np.full_like(labels_full, -1)
        for g in gs:
            r = find(g)
            if r not in remap:
                remap[r] = nxt
                nxt += 1
            out[labels_full == g] = remap[r]
        return out

    # ---- SPLIT: forward-only per-part tracking ----
    def _step_split(self, rgb, depth, mask, hyps, i):
        cfg = self.cfg
        a = self.assign
        H, W = self.H, self.W
        mask_b = mask > 0

        # what each part currently covers, so the parts can exclude each other
        occ = [a.occupancy(p.pose, p.weights, self.K, H, W, dilate=2)
               if p.pose is not None else None for p in self.parts]

        # how much observed object surface no part explains -- the direction the
        # per-gaussian energy is blind to, since it only sums over gaussians
        seen = mask_b & (depth > 0)
        cov = None
        for o in occ:
            if o is not None:
                o_np = o.cpu().numpy() if hasattr(o, "cpu") else o
                cov = o_np if cov is None else (cov | o_np)
        self.unmodelled = (float((seen & ~cov).sum()) / max(1, int(seen.sum()))
                           if cov is not None else 1.0)

        R = a.residuals(hyps, self.K, H, W, depth, obs_mask=mask)
        Pm_t = a.soft_membership(R)
        Pm = Pm_t.detach().cpu().numpy()
        self.last_hyp_set = list(hyps)
        self.last_share = (Pm_t.sum(dim=1)
                           / max(float(Pm_t.sum()), 1e-6)).cpu().numpy()

        for j, part in enumerate(self.parts):
            w = part.weights
            # the object mask minus what the OTHER parts hold
            om = mask
            if cfg.exclusive_mask:
                others = None
                for k2, o in enumerate(occ):
                    if k2 == j or o is None:
                        continue
                    others = o if others is None else (others | o)
                if others is not None:
                    if occ[j] is not None:
                        others = others & (~occ[j])
                    cand_m = mask_b & (~others)
                    if cand_m.sum() >= 200:
                        om = cand_m.astype(np.uint8)

            cands = []
            score = (Pm * w[None, :Pm.shape[1]]).sum(axis=1)
            k = int(np.argmax(score))
            cands.append(hyps[k] if score[k] > 0 else part.pose)
            cands.append(part.pose)
            if part.d_pose is not None:
                for s in (0.5, 1.0, 1.5, 2.0):
                    cands.append(se3_pow(part.d_pose, s) @ part.pose)
            cands.extend(np.asarray(T) for T in hyps)

            # The global hypotheses are dominated by the biggest surface, so a
            # thin fast-rotating part needs one fitted to its own tracks.
            tidx = part.track_idx
            if tidx is not None and tidx.size >= 6 and self.reg is not None:
                cu, cok, cvi = self.cur_tracks
                tidx = tidx[tidx < len(cok)]        # see _draw_tracks
                sel_t = tidx[self.anchor_ok[tidx] & cok[tidx] & cvi[tidx]]
                if sel_t.size >= 6:
                    c = self.reg._RANSAC(p0=self.anchor_xyz[sel_t],
                                         tgt_pcd=cu[sel_t], w=None,
                                         remaining=np.ones(sel_t.size, bool),
                                         init_pose=None)
                    if c is not None:
                        cands.append(c["T"])

            # the joint reduces the pose to a scalar, so sweep it
            jm = part.joint
            Tp = self.parts[part.parent].pose if jm is not None else None
            if jm is not None and jm.kind is not None and Tp is not None and part.joint_values:
                vs = part.joint_values
                span = max(max(vs) - min(vs), 1e-3)
                pred = vs[-1] + (vs[-1] - vs[-2] if len(vs) > 1 else 0.0)
                full = (np.linspace(-np.pi, np.pi, 121) if jm.kind == "revolute"
                        else np.linspace(-0.6, 0.6, 121))
                grid = np.concatenate(
                    [pred + np.linspace(-1.0, 1.0, 41) * max(span, 0.1), full])
                Tg = [Tp @ jm.at(float(v)) for v in grid]
                eg = a.fit_energy_batch(Tg, w, self.K, H, W, depth,
                                        obs_mask=om, r_max=cfg.track_rmax,
                                        outside_w=cfg.outside_w)
                for oi in np.argsort(eg)[:3]:
                    cands.append(Tg[int(oi)])

            best_T, best_e = None, float("inf")
            for Tc in cands:
                # a part cannot teleport between two consecutive frames
                if cfg.max_step > 0 and part.pose is not None:
                    if np.linalg.norm(np.asarray(Tc)[:3, 3] - part.pose[:3, 3]) > cfg.max_step:
                        continue
                # visibility from the candidate itself: only the part's own
                # front face is comparable against the observed depth
                vz = a.self_visible(Tc, w, self.K, H, W) if cfg.visibility else None
                Tr = a.refine_pose(Tc, w, self.K, H, W, depth, iters=8,
                                   huber=0.04, obs_mask=om, visible=vz)
                e = a.fit_energy(Tr, w, self.K, H, W, depth, obs_mask=om,
                                 r_max=cfg.track_rmax, visible=vz,
                                 outside_w=cfg.outside_w)
                if e < best_e:
                    best_T, best_e = Tr, e
            if best_T is None:
                continue
            vz = a.self_visible(best_T, w, self.K, H, W) if cfg.visibility else None
            Tf = a.refine_pose(best_T, w, self.K, H, W, depth, iters=8,
                               huber=0.006, obs_mask=om, visible=vz)
            ef = a.fit_energy(Tf, w, self.K, H, W, depth, obs_mask=om,
                              r_max=cfg.track_rmax, visible=vz,
                              outside_w=cfg.outside_w)
            if ef < best_e:
                best_T, best_e = Tf, ef

            if cfg.graph:
                if part.opt is None:
                    part.opt = self._new_graph()
                if part.opt is not None:
                    Tg = self._graph_pose(part, j, best_T, i)
                    # the graph only replaces the greedy pose when it still
                    # explains the depth at least as well
                    if Tg is not None:
                        eg = a.fit_energy(Tg, w, self.K, H, W, depth, obs_mask=om,
                                          r_max=cfg.track_rmax,
                                          outside_w=cfg.outside_w)
                        if eg <= best_e * 1.05:
                            best_T, best_e = Tg, eg
                            self.graph_used = getattr(self, "graph_used", 0) + 1

            part.d_pose = best_T @ np.linalg.inv(part.pose)
            part.pose, part.energy = best_T, best_e
            part.best_energy = min(part.best_energy, best_e)

            # feed the joint from poses the tracker trusts, and refit as it learns
            if jm is not None and Tp is not None and best_e < cfg.joint_gate:
                A = np.linalg.inv(Tp) @ best_T
                jm.add(A)
                if jm.kind is None or len(jm.A) % 4 == 0:
                    if jm.fit():
                        part.joint_values = [jm.value_of(x) for x in jm.A]
                elif jm.kind is not None:
                    part.joint_values.append(jm.value_of(A))

        # ---- keyframes, after Point2Pose's own sampling criterion ----
        for j, part in enumerate(self.parts):
            if part.view_dirs is None:
                part.view_dirs = [np.array([0., 0., 1.])]
            u = part.pose[:3, :3] @ part.view_dirs[0]
            # a viewpoint already covered is never sampled twice; comparing only
            # against the LAST keyframe re-seeds forever once a part oscillates
            ang = np.degrees(np.arccos(np.clip(
                [u @ v for v in part.view_dirs], -1.0, 1.0)))
            new_view = bool(np.all(ang >= cfg.key_angle_deg))
            # or the part still shows a surface but has run out of live tracks
            n_live = 0
            if part.track_idx is not None:
                _, cok, cvi = self.cur_tracks
                ti = part.track_idx[part.track_idx < len(cok)]
                n_live = int((cok[ti] & cvi[ti]).sum()) if ti.size else 0
            starved = (n_live < cfg.key_min_pts
                       and occ[j] is not None and occ[j].sum() > 200)
            if not (new_view or starved):
                continue
            if not self._grow_ok(part) and not starved:
                continue        # do not anchor through a pose that has slipped
            if len(self.anchor_xyz) >= cfg.key_max_points:
                continue
            if self._reseed(part, rgb, depth, mask, i, occ[j]):
                part.view_dirs.append(u)
                # grow the dense model at the same moment: the surface that
                # justified new tracks is the surface the model is missing
                self._grow(rgb, depth, mask)

        # ---- carve away surface the sensor sees straight through ----
        # Residuals mark a gaussian projecting outside the mask as UNOBSERVED,
        # which is exactly where a ghost copy of a neighbouring part lands, so
        # it is never questioned. Free-space carving asks the opposite: if the
        # sensor reports something BEHIND where this part claims surface, the
        # space between is empty and the claim is wrong. A hand in front gives
        # the other sign and is left alone, which is EM-Fusion's unoccluded-only
        # update stated per gaussian.
        if cfg.prune and self.parts and i % cfg.prune_every == 0:
            dt = torch.as_tensor(depth, dtype=torch.float32, device=a.device)
            gmm = self.cloud.means
            Kt = torch.as_tensor(self.K, dtype=torch.float32, device=a.device)
            n = len(self.cloud)
            free = np.zeros(n, bool)
            for j, p in enumerate(self.parts):
                T = torch.as_tensor(p.pose, dtype=torch.float32, device=a.device)
                q = gmm @ T[:3, :3].T + T[:3, 3]
                z = q[:, 2]
                u = (q[:, 0] * Kt[0, 0] / z + Kt[0, 2]).round().long()
                v_ = (q[:, 1] * Kt[1, 1] / z + Kt[1, 2]).round().long()
                ok = (z > 1e-3) & (u >= 0) & (u < W) & (v_ >= 0) & (v_ < H)
                zo = torch.zeros_like(z)
                zo[ok] = dt[v_[ok].clamp(0, H - 1), u[ok].clamp(0, W - 1)]
                carve = (ok & (zo > 0) & (zo > z + cfg.prune_margin)).cpu().numpy()
                free |= carve & (p.weights[:n] > 0.5)
            v = getattr(self, "_prune_votes", np.zeros(0, np.int16))
            if len(v) < n:
                v = np.concatenate([v, np.zeros(n - len(v), np.int16)])
            v[:n] = np.where(free, v[:n] + 1, 0)
            self._prune_votes = v
            go = np.where(v[:n] >= cfg.prune_votes)[0]
            if go.size:
                for p in self.parts:
                    p.weights[go] = 0.0
                v[go] = 0
                self.pruned_total = getattr(self, "pruned_total", 0) + int(go.size)
                gm2 = self.cloud.means.detach().cpu().numpy()
                for p in self.parts:
                    p.box = self._fit_box(p.weights[:len(gm2)] > 0.5)

        # ---- keep accumulating, so a part can still split later ----
        # The hypotheses that matter now are the parts' own poses; feeding the
        # global set alone would keep voting for the pre-split motion.
        if cfg.resplit:
            hy = [np.asarray(p.pose) for p in self.parts if p.pose is not None]
            for T in hyps:
                if len(hy) >= cfg.max_hyp:
                    break
                hy.append(np.asarray(T))
            hy = hy[:cfg.max_hyp]
            a.step_pointwise(hy, self.K, H, W, depth, obs_mask=mask)
            Pm2 = a.soft_membership(
                a.residuals(hy, self.K, H, W, depth, obs_mask=mask))
            self.pose_log.append({"frame": i, "T": hy,
                                  "P": Pm2.detach().cpu().numpy()})
            if len(self.pose_log) > cfg.pose_log_max:
                self.pose_log.pop(0)
            since = i - getattr(self, "last_split_frame", 0)
            if since >= cfg.resplit_wait and i % cfg.regroup_every == 0:
                t0 = time.perf_counter()
                prev_cov = float(getattr(self, "split_info", {}).get("coverage", 0.0))
                cur_lab = np.full(len(self.cloud), -1, dtype=int)
                for j2, p2 in enumerate(self.parts):
                    cur_lab[p2.weights[:len(cur_lab)] > 0.5] = j2
                cur_ratio = self._grouping_score(cur_lab)[0]
                self.cur_ratio = cur_ratio
                self._try_split(min_groups=len(self.parts) + 1,
                                max_groups=len(self.parts) + 1,
                                min_ratio=max(cfg.resplit_min_ratio,
                                              cfg.resplit_gain * cur_ratio),
                                min_cov=cfg.resplit_cov_frac * prev_cov,
                                require_relative_motion=True)
                self.last_split_ms = (time.perf_counter() - t0) * 1e3

        # ---- online part modelling ----
        if cfg.grow_every > 0 and i % cfg.grow_every == 0:
            self._grow(rgb, depth, mask)

    def _new_graph(self):
        """One ISAM2 graph per part, or None if GTSAM is unavailable."""
        try:
            from point2pose.modules.optimizer.isam2_optimizer import ISAM2Optimizer
        except Exception as exc:
            if not getattr(self, "_graph_warned", False):
                print(f"[stream] pose graph disabled ({exc})")
                self._graph_warned = True
            return None
        return ISAM2Optimizer({"relinearize_threshold": self.cfg.graph_relin,
                               "relinearize_skip": 1,
                               "prior_noise_param": [self.cfg.graph_prior] * 6})

    def _graph_pose(self, part, j, T, i):
        """Smooth this part's pose through its own graph over its own tracks."""
        from point2pose.data_types.object_frame_data import ObjectFrameData
        cur, cok, cvi = self.cur_tracks
        ti = part.track_idx
        n3 = np.zeros((0, 3), np.float32)
        idx = np.zeros(0, np.int64)
        inl = np.zeros(0, bool)
        res = np.zeros(0, np.float32)
        if ti is not None and ti.size:
            ti = ti[ti < len(cok)]
            sel = ti[cok[ti] & cvi[ti] & self.anchor_ok[ti]]
            if sel.size:
                pred = self.anchor_xyz[sel] @ T[:3, :3].T + T[:3, 3]
                d = np.linalg.norm(pred - cur[sel], axis=1)
                n3, idx = cur[sel].astype(np.float32), sel.astype(np.int64)
                inl = d < self.cfg.inlier_thres * 3
                res = np.maximum(d, 1e-4).astype(np.float32)
        data = ObjectFrameData(
            obj_id=int(j), frame_id=int(i), intrinsics=self.K,
            pose=np.asarray(T, dtype=np.float64),
            rel_pose=None if part.d_pose is None else np.asarray(part.d_pose,
                                                                 dtype=np.float64),
            visible_pts_2d=np.zeros((0, 2), np.float32),
            visible_pts_2d_idx=np.zeros(0, np.int64),
            visible_uncertainties=np.zeros(0, np.float32),
            reg_cur_3d=n3, reg_cur_3d_idx=idx,
            reg_valid_idx=np.arange(len(idx), dtype=np.int64),
            reg_inliers=inl, reg_residuals=res, reg_uncertainties=res)
        try:
            out = part.opt.optimize(data)
        except Exception:
            return None
        part.opt_n += 1
        return None if out is None else np.asarray(out.pose_optimized)

    def _tracks_on(self, gm, w):
        """Which sparse tracks sit on this part, by nearest gaussian in the
        anchor frame. The part then has its own hypothesis generator."""
        mem = gm[w > 0.5]
        if mem.shape[0] < 10 or self.anchor_xyz.shape[0] == 0:
            return None
        from scipy.spatial import cKDTree
        dist, _ = cKDTree(mem).query(self.anchor_xyz, k=1)
        thr = max(0.01, float(np.percentile(dist, 20)) * 2.0)
        return np.where(dist < thr)[0]

    def _fit_box(self, sel):
        """Oriented box around a set of gaussians, in the anchor frame."""
        try:
            from experiments.articulated.demo_part_discovery import fit_oriented_box
        except Exception:
            return None
        gm = self.cloud.means.detach().cpu().numpy()
        q = gm[sel]
        if q.shape[0] < 20:
            return None
        try:
            # trim the outer few percent: one stray grown point inflates a box
            d = np.linalg.norm(q - np.median(q, axis=0), axis=1)
            return fit_oriented_box(q[d <= np.percentile(d, 97)])
        except Exception:
            return None

    def render(self, bgr, palette=None, show_tracks=True):
        """Draw the current state onto a BGR image."""
        import cv2
        from point2pose.utils.visualization import draw_oriented_3d_box

        pal = palette or [(60, 140, 235), (200, 120, 40), (70, 180, 90),
                          (200, 80, 200), (60, 200, 200)]
        gm = self.cloud.means.detach().cpu().numpy()
        K = self.K
        out = bgr

        def splat(pts3, color, radius=1):
            z = pts3[:, 2]
            ok = z > 1e-3
            if not ok.any():
                return
            u = (pts3[ok, 0] * K[0, 0] / z[ok] + K[0, 2]).astype(int)
            v = (pts3[ok, 1] * K[1, 1] / z[ok] + K[1, 2]).astype(int)
            good = (u >= 0) & (u < out.shape[1]) & (v >= 0) & (v < out.shape[0])
            for uu, vv in zip(u[good], v[good]):
                cv2.circle(out, (uu, vv), radius, color, -1)

        if self.state == self.RIGID:
            T = self.whole_pose
            splat(gm @ T[:3, :3].T + T[:3, 3], (150, 150, 150))
            if self.whole_box is not None:
                try:
                    out = draw_oriented_3d_box(K, out, T, self.whole_box,
                                               line_color=(200, 200, 200), linewidth=2)
                except Exception:
                    pass
            if show_tracks:
                self._draw_tracks(out, None, pal)
            return out

        for j, p in enumerate(self.parts):
            col = pal[j % len(pal)]
            sel = p.weights[:gm.shape[0]] > 0.5
            T = p.pose
            splat(gm[sel] @ T[:3, :3].T + T[:3, 3], col)
            if p.box is not None:
                try:
                    out = draw_oriented_3d_box(K, out, T, p.box,
                                               line_color=col, linewidth=2)
                except Exception:
                    pass
            # pose axes at the part centroid
            c = gm[sel].mean(0) if sel.any() else np.zeros(3)
            org = T[:3, :3] @ c + T[:3, 3]
            for ax, acol in zip(np.eye(3) * 0.05,
                                [(60, 60, 235), (60, 220, 60), (235, 160, 60)]):
                tip = T[:3, :3] @ (c + ax) + T[:3, 3]
                if org[2] > 1e-3 and tip[2] > 1e-3:
                    p0 = (int(org[0] * K[0, 0] / org[2] + K[0, 2]),
                          int(org[1] * K[1, 1] / org[2] + K[1, 2]))
                    p1 = (int(tip[0] * K[0, 0] / tip[2] + K[0, 2]),
                          int(tip[1] * K[1, 1] / tip[2] + K[1, 2]))
                    cv2.arrowedLine(out, p0, p1, acol, 2, cv2.LINE_AA, tipLength=0.3)

        # the estimated joint axis, once per pair
        drawn = set()
        for j, p in enumerate(self.parts):
            jm = p.joint
            if jm is None or jm.kind is None or p.parent == j:
                continue
            key = frozenset((j, p.parent))
            if key in drawn:
                continue
            drawn.add(key)
            Tp = self.parts[p.parent].pose
            sel = p.weights[:gm.shape[0]] > 0.5
            c = gm[sel].mean(0) if sel.any() else np.zeros(3)
            # the fitted point is only determined up to a slide along the axis
            base = (jm.point + jm.axis * float((c - jm.point) @ jm.axis)
                    if jm.kind == "revolute" else c)
            seg = np.stack([base - jm.axis * 0.10, base + jm.axis * 0.10])
            q = seg @ Tp[:3, :3].T + Tp[:3, 3]
            if (q[:, 2] > 1e-3).all():
                pu = (q[:, 0] * K[0, 0] / q[:, 2] + K[0, 2]).astype(int)
                pv = (q[:, 1] * K[1, 1] / q[:, 2] + K[1, 2]).astype(int)
                cv2.line(out, (pu[0], pv[0]), (pu[1], pv[1]), (255, 255, 255), 4, cv2.LINE_AA)
                cv2.line(out, (pu[0], pv[0]), (pu[1], pv[1]), (30, 30, 30), 2, cv2.LINE_AA)
                cv2.putText(out, jm.kind, (pu[1] + 6, pv[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
        if show_tracks:
            self._draw_tracks(out, self.parts, pal)
        return out

    def _grow_ok(self, part):
        """Is this part explaining the surface well enough to anchor new data?"""
        cfg = self.cfg
        if part.energy > cfg.grow_gate_abs:
            return False
        if not np.isfinite(part.best_energy):
            return True
        return part.energy <= cfg.grow_gate_rel * part.best_energy

    def _grow(self, rgb, depth, mask):
        """Add gaussians for object surface no part explains yet."""
        cfg, a = self.cfg, self.assign
        poses = [p.pose for p in self.parts]
        ws = [p.weights for p in self.parts]
        ok_g = [self._grow_ok(p) for p in self.parts]
        n_new, owner = a.grow_parts(rgb, depth, self.K, mask, poses, ws,
                                    stride=self.gauss_stride,
                                    max_new=cfg.grow_max, grow_ok=ok_g,
                                    max_total=cfg.gauss_max)
        self.grow_calls = getattr(self, "grow_calls", 0) + 1
        self.grow_blocked = getattr(self, "grow_blocked", 0) + int(not any(ok_g))
        self.grown_total = getattr(self, "grown_total", 0) + int(n_new or 0)
        if not n_new:
            return 0
        for j, p in enumerate(self.parts):
            p.weights = np.concatenate(
                [p.weights, (owner == j).astype(np.float32)])
        gm = self.cloud.means.detach().cpu().numpy()
        for p in self.parts:
            p.box = self._fit_box(p.weights > 0.5)
            # the model grew, so which tracks sit on this part can change
            p.track_idx = self._tracks_on(gm, p.weights)
        return n_new

    def _reseed(self, part, rgb, depth, mask, i, occ_j):
        """Sample fresh tracks on the surface this part shows now.

        A part that has turned loses the tracks it was seeded with -- measured
        offline, that is where a thin fast-rotating part's pose gave out. New
        points are anchored through the part's current pose, exactly as grown
        gaussians are.
        """
        from point2pose.data_types.frame import Frame

        cfg = self.cfg
        region = (occ_j & (mask > 0) & (depth > 0.05)) if occ_j is not None else None
        if region is None or region.sum() < 200:
            return False
        pts = sample_superpoint(self.sampler, rgb, depth,
                                region.astype(np.uint8), self.K, cfg.key_points)
        if pts.shape[0] < 8:
            return False
        xyz, ok = lift(pts, depth, self.K)
        if ok.sum() < 8:
            return False

        f = Frame(id=i, rgb=rgb, depth=depth, intrinsics=self.K)
        try:
            new_idx = self.tracker.add_query_points(f, pts)
        except Exception as exc:
            print(f"[stream] re-seed failed: {exc}")
            return False
        new_idx = np.asarray(new_idx).reshape(-1)

        T = part.pose
        anchor = (xyz - T[:3, 3]) @ T[:3, :3]        # camera -> anchor frame
        self.anchor_xyz = np.concatenate([self.anchor_xyz, anchor.astype(np.float32)])
        self.anchor_ok = np.concatenate([self.anchor_ok, ok])
        # every earlier frame in the window predates these tracks
        for j2 in list(self.past):
            p_, o_, v_ = self.past[j2]
            pad = np.zeros((len(new_idx), 3), np.float32)
            self.past[j2] = (np.concatenate([p_, pad]),
                             np.concatenate([o_, np.zeros(len(new_idx), bool)]),
                             np.concatenate([v_, np.zeros(len(new_idx), bool)]))
        part.track_idx = (new_idx if part.track_idx is None
                          else np.concatenate([part.track_idx, new_idx]))
        print(f"[stream] frame {i}: re-seeded {len(new_idx)} tracks on a part "
              f"({len(self.anchor_xyz)} total)")
        return True

    @staticmethod
    def fit_panel(panel, width):
        """Scale a panel down to `width` and pad the remainder. Six parts make
        the strip wider than the view, and np.full then gets a negative size."""
        import cv2
        if panel.shape[1] > width:
            h = max(1, int(panel.shape[0] * width / panel.shape[1]))
            panel = cv2.resize(panel, (width, h))
        if panel.shape[1] == width:
            return panel
        pad = np.full((panel.shape[0], width - panel.shape[1], 3),
                      (32, 30, 28), np.uint8)
        return np.hstack([panel, pad])

    def hypothesis_panel(self, depth, mask=None, width=200, r_max=0.05):
        """What each motion explains: its RGB render on top, its residual below.

        Before the split there is one body, so every hypothesis is drawn with the
        whole cloud. Afterwards each part owns its gaussians and is drawn with
        those alone -- rendering the whole cloud under a part's pose shows a
        disagreement that belongs to the other parts, not to this one.
        """
        import cv2

        if self.state == self.SPLIT and self.parts:
            items = [(f"p{j}", p.pose, torch.as_tensor(
                        p.weights[:len(self.cloud)] > 0.5, device=self.assign.device))
                     for j, p in enumerate(self.parts)]
        else:
            hs = getattr(self, "last_hyp_set", None)
            if not hs:
                return None
            share = getattr(self, "last_share", np.zeros(len(hs)))
            items, seen = [], []
            for k, T in enumerate(hs):
                if any(np.allclose(T, U, atol=1e-6) for U in seen):
                    continue
                seen.append(np.asarray(T))
                items.append((f"h{k} {100 * share[k]:.0f}%", T, None))
        obs = torch.as_tensor(depth, dtype=torch.float32,
                              device=self.assign.device)
        mk = (torch.as_tensor(mask > 0, device=obs.device)
              if mask is not None else torch.ones_like(obs, dtype=torch.bool))
        # Render everything first: "no model here" is only meaningful against
        # the union of all parts, not against one part that never owned it.
        shots = [(label,) + self.cloud.render(T, self.K, depth.shape[0],
                                              depth.shape[1], subset=subset)
                 for label, T, subset in items]
        any_drawn = torch.zeros_like(obs, dtype=torch.bool)
        for _, _, _, alpha in shots:
            any_drawn |= alpha > 0.3
        gap = (~any_drawn & mk & (obs > 0)).cpu().numpy()

        tiles = []
        for label, col, dep_r, alpha in shots:
            a_np = (alpha > 0.3).cpu().numpy()

            rgb_t = (col.clamp(0, 1) * 255).byte().cpu().numpy()
            rgb_t = cv2.cvtColor(rgb_t, cv2.COLOR_RGB2BGR)
            rgb_t[~a_np] = (32, 30, 28)

            # Four states, not two. Painting everything the model does not cover
            # as background reads as "no error" when it means "not measured".
            drawn = alpha > 0.3
            vis = drawn & (obs > 0) & mk
            res = (dep_r - obs).abs().clamp(max=r_max) / r_max
            err = cv2.applyColorMap(
                (res.cpu().numpy() * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
            err[~vis.cpu().numpy()] = (32, 30, 28)
            # model is here but the live mask says the object is not
            err[(drawn & ~mk).cpu().numpy()] = (200, 60, 200)
            # object is here and no part at all explains it
            err[gap] = (95, 95, 95)

            h = int(rgb_t.shape[0] * width / rgb_t.shape[1])
            rgb_t = cv2.resize(rgb_t, (width, h))
            err = cv2.resize(err, (width, h))
            cv2.putText(rgb_t, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)
            if not tiles:
                cv2.putText(err, f"unmodelled {100 * gap.sum() / max(1, int((mk & (obs > 0)).sum())):.0f}%",
                            (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (230, 230, 230), 1, cv2.LINE_AA)
            tiles.append(np.vstack([rgb_t, err]))
        if not tiles:
            return None
        strip = np.hstack(tiles)
        cv2.putText(strip, "blue-red residual | magenta model-outside-mask | "
                           "grey object-explained-by-no-part",
                    (6, strip.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (190, 190, 190), 1, cv2.LINE_AA)
        return strip

    def _draw_tracks(self, out, parts, pal):
        """The sparse point tracks, coloured by the part they belong to.

        Worth seeing: these are what the hypotheses are fitted from, and after the
        split each part fits its own RANSAC from the tracks that sit on it. A part
        whose tracks have gone invisible is a part whose pose is running on the
        dense model alone.
        """
        import cv2

        t2d = getattr(self, "cur_tracks_2d", None)
        if t2d is None:
            return
        _, ok, vis = self.cur_tracks
        owner = np.full(len(t2d), -1, dtype=int)
        if parts:
            for j, p in enumerate(parts):
                if p.track_idx is not None:
                    # tracks re-seeded this frame are not in the tracker's output
                    # until the next one
                    ti = p.track_idx[p.track_idx < len(t2d)]
                    owner[ti] = j

        for i in range(len(t2d)):
            x, y = int(t2d[i, 0]), int(t2d[i, 1])
            if not (0 <= x < out.shape[1] and 0 <= y < out.shape[0]):
                continue
            col = pal[owner[i] % len(pal)] if owner[i] >= 0 else (190, 190, 190)
            if vis[i] and ok[i]:
                cv2.circle(out, (x, y), 3, (20, 20, 20), -1)
                cv2.circle(out, (x, y), 2, col, -1)
            else:
                # hollow: the tracker says this point is occluded or lost
                cv2.circle(out, (x, y), 3, col, 1)

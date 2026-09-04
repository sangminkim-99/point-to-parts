"""Sparse-only articulated tracker: the method with nothing added.

Steps 1-3 of the plan and no more -- point tracks, a rigid fit per part, a split
when one rigid motion stops explaining the residual, and a screw axis from the
relative pose trajectory. No gaussians, no rendering, no dense energy. This is
the baseline the dense machinery has to beat, measured the same way.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from scipy.spatial.transform import Rotation as _R

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
    inlier_thres: float = 0.02
    min_inliers: int = 3
    ransac_iters: int = 200
    num_pips_iter: int = 1
    tapir_res: int = 512
    # A split is proposed when a MINORITY of points stops following the fit.
    # The median residual cannot see that: a second part is by definition the
    # smaller set, so the median stays with the majority and never rises.
    # Thresholds in metres cannot see a wiggle. Fifty points displaced 1 cm the
    # same way is overwhelming evidence against one rigid body even though no
    # single point moved far, so the trigger and the separation test are both
    # measured in units of the depth noise the fit itself reveals.
    split_out_frac: float = 0.12        # share of points beyond the inlier band
    split_out_band: float = 3.0         # x the estimated noise sigma
    sigma_floor: float = 0.002          # metres; a RealSense at 0.5 m
    sigma_ceil: float = 0.02
    split_sep_sigma: float = 4.0        # group displacement, in sigma
    split_res: float = 0.05             # metres of median residual, as a backstop
    split_frames: int = 3               # consecutive frames above either
    min_part_pts: int = 6
    # The two motions must actually differ, or a noisy fit splits a rigid body:
    # foldingrule01 ran to 6 parts against a ground truth of 3.
    split_gain: float = 0.7             # new residual must be this x the old
    # A split adds a second 6-DoF body, so it must pay for six parameters before
    # it is worth making. Without this the co-association path has no test that
    # the split explains the points better at all, and opening and closing a
    # joint repeatedly shaves off a new part every time.
    split_bic: bool = True
    split_bic_margin: float = 0.0       # extra BIC the split must win by
    # Points seeded together are anchored through one pose, so an error in that
    # pose is written into all of them as a single rigid offset -- which is
    # indistinguishable from a second part, and spatially coherent, so the
    # clustering finds it every time. Measured on RBO: three of five late splits
    # separated the tracks exactly along seeding batches.
    # Provenance is the wrong test: re-seeded points that move WITH the part are
    # the part. What separates an anchoring artefact from a joint is that the
    # artefact is a constant offset -- the two groups never move relative to
    # each other -- while a joint's relative transform varies. So parts are
    # allowed to form and are merged back when their joint never opens.
    cohort_veto: bool = False
    cohort_pure: float = 0.8            # share of one batch that counts as pure
    merge_rigid: bool = True
    merge_every: int = 10               # frames between merge checks
    merge_min_obs: int = 12             # relative poses needed to judge
    # A real joint beats the rigid model by thousands of BIC; a jittery pose on
    # a small part beats it by about nine. Kass & Raftery call dBIC > 10 very
    # strong evidence, and that is the line between a part and an artefact.
    merge_bic: float = 10.0
    # Residual evidence cannot separate a joint from correlated tracking error;
    # the smoothness of q(t) can, because a real joint is driven and error is
    # white. Measured on synthetic data: real joints 1.00, pure jitter 0.00.
    merge_smooth: float = 0.15
    # A hinge is a physical thing: it lives in or on the object. Used as a MAP
    # prior on the axis, scaled by the object's own radius. Proposal generators
    # assume this (H-SAUR, Real2Code); using it to REJECT a joint is new.
    # A hinge is ON the object, so 0.2 of its radius, not 0.5. Measured on a
    # 20 cm object lifted 20 cm with the rotation an under-constrained pose fit
    # adds: at 0.5 the fit prefers a revolute joint pivoting 1.3 m away, at 0.2
    # it does not. Real hinges sit 0-4 cm from the surface and pay nothing.
    axis_prior: float = 0.2             # x the object radius; 0 disables
    # Stage B: a single frame's RANSAC split is a coin toss on noisy depth. What
    # marks a real part is the SAME point subset backing a different motion frame
    # after frame, so co-association is accumulated and clustered instead.
    persist: bool = True
    co_hyp: int = 3                     # motions proposed per frame
    co_tau: float = 0.01                # metres, softmax temperature
    co_min_seen: float = 2.5            # weighted frames a pair must co-occur
    co_gap: float = 0.15                # affinity gap the two clusters need
    # A new part inherits no history: the affinity that split its parent would
    # otherwise split it again the moment it is allowed to, and cabinet03 ran to
    # 6 parts against a ground truth of 3.
    part_settle: int = 20               # frames a part must live before splitting
    # Co-association covers 3/3 and 2/2 GT parts where it fires but stays silent
    # on laptop02 and cardboardbox02, where one frame's RANSAC did split. So the
    # frame path is kept, but only once the part has been inconsistent for far
    # longer than co-association needed.
    frame_fallback: bool = True
    frame_after: int = 5                # x split_frames before one frame decides
    ambiguous_band: float = 1.5         # x inlier_thres: points in the joint gap
    min_frames_before_split: int = 8
    resplit_wait: int = 12
    max_parts: int = 6
    joint_gate: float = 0.02
    # Step 3b: once a joint is confident the part has one degree of freedom, not
    # six. Searching that scalar instead is what survives occlusion.
    joint_track: bool = True
    joint_min_obs: int = 12
    # The object is "controllable" once a joint is identified well enough to
    # command a target q; that moment, not the runtime, is the online claim.
    joint_conf: float = 0.6             # on type x axis alone
    joint_exc: float = 0.5              # and enough of the range seen
    joint_tol: float = 1.3              # x the free fit's residual before falling back
    joint_grid: int = 61
    # A thin object leaves most SuperPoint picks on invalid depth -- pliers01
    # kept 10 of 100 -- and a part can never split below 2 * min_part_pts.
    top_up: bool = True
    min_live: int = 24                  # per part, before new points are sought
    reseed_points: int = 24
    reseed_every: int = 6
    max_points: int = 400
    # Re-seeding anchors new tracks through the pose held at that moment, so a
    # pose that is even slightly off writes the error into them; they then
    # disagree with the older tracks, the outlier share rises and the part
    # splits again. Fresh tracks therefore carry no split evidence until they
    # have been watched, and no part re-seeds through a pose it cannot trust.
    # A track seeded inside a part, anchored through that part's pose, IS that
    # part -- we put it there. Starting its co-association at zero throws that
    # away and lets a few noisy frames carve it back out. It starts instead with
    # recorded agreement with its part, which evidence then has to overcome.
    # A new track's part is a question about MOTION, and it was being answered
    # by position -- sampled inside a part's box, anchored through that part's
    # pose at that instant, so any error in it was written in permanently. A
    # track belongs to part j when inv(T_j(t)) x(t) stays put, which needs no
    # anchor at all and decides itself over several frames.
    pending: bool = True
    pend_min_obs: int = 8               # observations before a track is placed
    pend_max_age: int = 45              # frames before an undecided track is dropped
    pend_margin: float = 1.8            # x better than the runner-up to claim it
    pend_sigma: float = 3.0             # x noise the winner must still be under
    # Motion evidence alone cannot tell "moves with" from "is wrong with":
    # tracking error is correlated in time, so it fakes agreement. Where the
    # point LANDS is a second, independent source -- a part's own tracked points
    # are its observed surface, so a new point on that surface, at that depth,
    # belongs to it. Reprojection and temporal consistency have to agree.
    reproject: bool = True
    reproj_px: float = 24.0             # how near the part's surface, in pixels
    reproj_depth: float = 0.03          # and in metres
    seed_prior: float = 6.0             # weighted frames of assumed agreement
    track_grace: int = 12               # frames before a track may vote
    reseed_max_resid: float = 0.012     # metres; pose must explain the old ones
    reseed_gap: int = 5                # frames between re-seeds of one part
    # Point2Pose's own criterion -- sample a viewpoint once, when it first turns
    # towards the camera -- implemented and measured to be worse here, so off.
    # It re-seeds every 15 degrees of rotation, and every seeding batch shares
    # one anchor pose whose error becomes a coherent fake part: pliers01 2 -> 3
    # parts, cardboardbox02 2 -> 3, laptop02's pose 64.5 -> 379.8 mm. Spinning a
    # rigid object in place is its worst case, which is where it was reported.
    key_view: bool = False
    key_angle_deg: float = 15.0
    sampler: str = "super_point_balanced"
    sampler_cell: int = -1
    sampler_nms: float = 0.0
    sampler_score_w: float = 0.20
    sampler_min_sep: float = 6.0
    # A track that leaves the object mask has walked onto the hand. On RBO
    # pliers that is what the second "rigid motion" was: the arm, not a jaw.
    mask_gate: bool = True
    mask_pad: int = 4
    mask_strikes: int = 5
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
    sigma: float = 0.004                # robust noise scale of this part's fit
    out_frac: float = 0.0
    n_mature: int = 0
    on_joint: bool = False
    view_dirs: list = None              # viewing directions already sampled
    box: object = None                  # oriented box in the anchor frame
    born: int = 0
    last_seed: int = -999
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
        self.mask = mask
        self.sampler = build_sampler(self.cfg)
        pts0 = sample_superpoint(self.sampler, rgb, depth, mask, self.K,
                                 self.cfg.n_points)
        f0 = Frame(id=0, rgb=rgb, depth=depth, intrinsics=self.K)
        self.tracker.add_query_points(f0, pts0)
        self.tracker.initialize(f0)
        pts0 = np.asarray(pts0, np.float32)
        if self.cfg.top_up:
            pts0 = self._top_up(pts0, depth, mask, self.cfg.n_points)
        self.pts0 = pts0
        self.anchor_xyz, self.anchor_ok = lift(pts0, depth, self.K)
        self.track_born = np.zeros(len(pts0), np.int32)
        self.pend = {}          # track index -> (n, sum, sumsq) per part
        self.parts = [NaivePart(idx=np.where(self.anchor_ok)[0])]
        self.parts[0].box = self._fit_box(self.parts[0])
        self.n = 1
        n = len(self.anchor_xyz)
        self.co_same = np.zeros((n, n), np.float32)
        self.co_seen = np.zeros((n, n), np.float32)
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
        from examples.multi_part.streaming import clean_mask
        cfg = self.cfg
        t0 = time.perf_counter()
        i = self.n
        self.n += 1
        # one mask for the whole frame: the gate, the re-seeding and the viewer
        # were each using a different one, so a track could be judged on-object
        # against pixels the sampler had already discarded
        mask = clean_mask(mask, depth)
        self.mask = mask

        tracks, _, vis = self.tracker.track_once(
            Frame(id=i, rgb=rgb, depth=depth, intrinsics=self.K))
        vis = vis.astype(bool)
        cur, cur_ok = lift(tracks, depth, self.K)
        if cfg.mask_gate:
            cur_ok = cur_ok & self._on_object(tracks, mask)
        self.cur = (cur, cur_ok, vis)
        self.cur_tracks_2d = tracks
        t_track = time.perf_counter()

        for p in self.parts:
            self._fit(p, cur, cur_ok, vis)
        if cfg.persist:
            for p in self.parts:
                self._accumulate(p, cur, cur_ok, vis)
        import os
        if os.environ.get("NAIVE_DEBUG") and i % 10 == 0:
            print("[naive] f%d " % i + "  ".join(
                f"p{j}: n={len(p.idx)} resid={p.resid*1000:.0f}mm "
                f"out={100*p.out_frac:.0f}% over={p.over}"
                for j, p in enumerate(self.parts)), flush=True)

        if i >= cfg.min_frames_before_split and len(self.parts) < cfg.max_parts \
                and i - getattr(self, "last_split", 0) >= cfg.resplit_wait:
            for j, p in enumerate(list(self.parts)):
                if self.n - p.born < cfg.part_settle:
                    continue
                if p.n_mature < 2 * cfg.min_part_pts:
                    continue
                if p.over >= cfg.split_frames and self._split(j, cur, cur_ok, vis):
                    self.last_split = i
                    break

        for j, p in enumerate(self.parts):
            if p.joint is None or j == p.parent or p.parent >= len(self.parts):
                continue
            if p.resid < cfg.joint_gate:
                # the BIC needs the real observation noise, and the fit itself
                # is the only thing that knows it
                p.joint.sigma = max(p.sigma, self.parts[p.parent].sigma)
                p.joint.axis_prior = cfg.axis_prior
                p.joint.geom = self._joint_geom(j, p)
                # rotation is only as well determined as the lever arm allows
                p.joint.sigma_r_floor = max(
                    self._rot_floor(p), self._rot_floor(self.parts[p.parent]))
                A = np.linalg.inv(self.parts[p.parent].pose) @ p.pose
                p.joint.add(A)
                if p.joint.kind is None or len(p.joint.A) % 4 == 0:
                    if p.joint.fit():
                        self._note_controllable(j, p, i)

        self.reproj = None
        if cfg.reproject and len(self.parts) > 1:
            self.reproj, _ = self._reproject_owner(tracks, depth)
        if cfg.pending:
            self._place_pending(i)

        if cfg.merge_rigid and len(self.parts) > 1 and i % cfg.merge_every == 0:
            self._merge_rigid(i)

        if cfg.top_up and i % cfg.reseed_every == 0:
            self._reseed(rgb, depth, mask, i)

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
        cfg = self.cfg
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

    def _on_object(self, t2, mask):
        """True where a track still sits inside the (padded) object mask."""
        import cv2
        cfg = self.cfg
        m = (mask > 0).astype(np.uint8)
        if cfg.mask_pad > 0:
            k = 2 * cfg.mask_pad + 1
            m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        u = np.clip(np.round(t2[:, 0]).astype(int), 0, m.shape[1] - 1)
        v = np.clip(np.round(t2[:, 1]).astype(int), 0, m.shape[0] - 1)
        inside = m[v, u] > 0
        st = getattr(self, "_strikes", np.zeros(0, np.int16))
        if len(st) < len(inside):
            st = np.concatenate([st, np.zeros(len(inside) - len(st), np.int16)])
        st[:len(inside)] = np.where(inside, 0, st[:len(inside)] + 1)
        self._strikes = st
        # a track that has been off the object for a while is gone for good
        dead = st[:len(inside)] >= cfg.mask_strikes
        self.dropped = int(dead.sum())
        return inside & ~dead

    @staticmethod
    def _top_up(pts, depth, mask, want):
        """Keep only points with valid depth.

        Refilling with random pixels was measured and removed: TAPIR cannot
        track an untextured point, and 90 random ones took the rigid fit's
        residual on RBO pliers from 2 mm to 78 mm.
        """
        _, ok = lift(pts, depth, np.eye(3))
        return pts[ok][:want]

    def _reseed(self, rgb, depth, mask, i):
        """New tracks for parts that have run out, inside their own region."""
        from point2pose.data_types.frame import Frame
        from examples.multi_part.streaming import sample_superpoint
        cfg = self.cfg
        cur, cok, cvi = self.cur
        t2 = self.cur_tracks_2d
        added = 0
        for j, p in enumerate(self.parts):
            idx = p.idx[p.idx < len(cok)]
            live = int((cok[idx] & cvi[idx]).sum()) if idx.size else 0
            if p.view_dirs is None:
                p.view_dirs = [np.array([0., 0., 1.])]
            u = p.pose[:3, :3] @ p.view_dirs[0]
            ang = np.degrees(np.arccos(np.clip(
                [float(u @ v) for v in p.view_dirs], -1.0, 1.0)))
            # a direction already covered is never sampled twice; comparing only
            # against the last one re-seeds forever once a part oscillates
            new_view = bool(cfg.key_view and np.all(ang >= cfg.key_angle_deg))
            if not new_view and live >= cfg.min_live:
                continue
            if len(self.anchor_xyz) >= cfg.max_points:
                continue
            if i - p.last_seed < cfg.reseed_gap:
                continue
            if p.resid > cfg.reseed_max_resid:
                continue        # do not anchor through a pose that has slipped
            m = mask.copy()
            if idx.size >= 3 and len(self.parts) > 1:
                q = t2[idx].astype(int)
                box = np.zeros_like(m)
                x0, y0 = np.clip(q.min(0) - 20, 0, None)
                x1, y1 = q.max(0) + 20
                box[y0:y1, x0:x1] = 1
                m = m * box
            new = np.asarray(sample_superpoint(self.sampler, rgb, depth, m,
                                                self.K, cfg.reseed_points),
                             np.float32)
            _, nok = lift(new, depth, self.K)
            new = new[nok][:cfg.reseed_points]
            if len(new) < 4:
                continue
            f = Frame(id=i, rgb=rgb, depth=depth, intrinsics=self.K)
            self.tracker.add_query_points(f, new)
            xyz, ok = lift(new, depth, self.K)
            base = len(self.anchor_xyz)
            n_new = len(new)
            self.anchor_xyz = np.concatenate(
                [self.anchor_xyz, np.zeros((n_new, 3), np.float32)])
            self.anchor_ok = np.concatenate(
                [self.anchor_ok, np.zeros(n_new, bool)])
            self.track_born = np.concatenate(
                [self.track_born, np.full(n_new, i, np.int32)])
            if cfg.pending:
                # nothing is claimed yet: the track waits for its own motion to
                # say which part it belongs to, and is anchored only then
                for k in np.where(ok)[0]:
                    self.pend[base + int(k)] = {"born": i, "obs": []}
            else:
                Ti = np.linalg.inv(p.pose)
                self.anchor_xyz[base:] = (xyz @ Ti[:3, :3].T
                                          + Ti[:3, 3]).astype(np.float32)
                self.anchor_ok[base:] = ok
                p.idx = np.concatenate([p.idx, base + np.where(ok)[0]])
            p.last_seed = i
            p.view_dirs.append(u)
            fresh = base + np.where(ok)[0]
            if cfg.seed_prior > 0 and fresh.size and not cfg.pending:
                self._grow_co(len(self.anchor_xyz))
                old_i = p.idx[(p.idx < len(self.anchor_xyz))
                              & ~np.isin(p.idx, fresh)]
                w = cfg.seed_prior
                for a_, b_ in ((fresh, old_i), (old_i, fresh), (fresh, fresh)):
                    if a_.size and b_.size:
                        self.co_same[np.ix_(a_, b_)] += w
                        self.co_seen[np.ix_(a_, b_)] += w
            added += int(ok.sum())
        if added:
            for p in self.parts:
                p.box = self._fit_box(p)
            print(f"[naive] frame {i}: re-seeded {added} tracks "
                  f"({len(self.pend)} awaiting a part, "
                  f"{sum(len(q.view_dirs or []) for q in self.parts)} views)")
        return added

    def _note_controllable(self, j, part, i):
        """First moment a joint is pinned down well enough to command."""
        c = part.joint.confidence()
        if c["conf"] < self.cfg.joint_conf or hasattr(self, "controllable"):
            return
        self.controllable = {
            "frame": int(i), "part": int(j), "kind": part.joint.kind,
            "conf": c["conf"], "axis_std_deg": c["axis_std_deg"],
            "span": c["span"], "split_frame": self.split_log[0][0]
            if self.split_log else None}
        span = (np.degrees(c["span"]) if part.joint.kind == "revolute"
                else c["span"] * 1000)
        unit = "deg" if part.joint.kind == "revolute" else "mm"
        print(f"[naive] CONTROLLABLE at frame {i}: part {j} {part.joint.kind}, "
              f"conf {c['conf']:.2f}, axis +-{c['axis_std_deg']:.1f} deg, "
              f"observed {span:.1f} {unit}")

    def _fit_joint(self, part, sel, cur):
        """Search the joint's scalar instead of a free SE(3), when it is known."""
        cfg = self.cfg
        jm = part.joint
        if not (cfg.joint_track and jm is not None and jm.kind is not None):
            return None
        if len(jm.A) < cfg.joint_min_obs or part.parent >= len(self.parts):
            return None
        c = jm.confidence()
        if not c.get("valid", True) or c["conf"] < cfg.joint_conf \
                or c["excitation"] < cfg.joint_exc:
            return None         # not a joint, or not yet known well enough
        Tp = self.parts[part.parent].pose
        if Tp is None:
            return None
        vs = [jm.value_of(A) for A in jm.A]
        span = max(max(vs) - min(vs), 1e-2)
        pred = vs[-1] + (vs[-1] - vs[-2] if len(vs) > 1 else 0.0)
        grid = np.concatenate([
            pred + np.linspace(-1.0, 1.0, cfg.joint_grid) * span,
            np.linspace(min(vs) - 0.2 * span, max(vs) + 0.2 * span, cfg.joint_grid)])
        P0 = self.anchor_xyz[sel]
        Q = cur[sel]
        best, bq = None, np.inf
        for v in grid:
            T = Tp @ jm.at(float(v))
            d = np.linalg.norm(P0 @ T[:3, :3].T + T[:3, 3] - Q, axis=1)
            q = float(np.median(d))
            if q < bq:
                best, bq = T, q
        return None if best is None else (best, bq)

    def _fit(self, part, cur, cur_ok, vis):
        """Rigid fit of this part's points from the anchor frame, RANSAC + SVD."""
        cfg = self.cfg
        # tracks added this frame have no measurement yet
        idx = part.idx[(part.idx < len(cur_ok)) & (part.idx < len(self.anchor_ok))]
        sel = idx[self.anchor_ok[idx] & cur_ok[idx] & vis[idx]]
        if sel.size < cfg.min_inliers or self.reg is None:
            part.over = 0
            return
        c = self.reg._RANSAC(p0=self.anchor_xyz[sel], tgt_pcd=cur[sel], w=None,
                             remaining=np.ones(sel.size, bool), init_pose=None)
        if c is None:
            part.over = 0
            return
        T_free = np.asarray(c["T"])
        d = np.linalg.norm(
            self.anchor_xyz[sel] @ T_free[:3, :3].T + T_free[:3, 3]
            - cur[sel], axis=1)
        part.pose = T_free
        jf = self._fit_joint(part, sel, cur)
        if jf is not None and jf[1] <= max(float(np.median(d)), 1e-4) * cfg.joint_tol:
            part.pose = jf[0]
            part.on_joint = True
            self.joint_frames = getattr(self, "joint_frames", 0) + 1
            d = np.linalg.norm(
                self.anchor_xyz[sel] @ part.pose[:3, :3].T + part.pose[:3, 3]
                - cur[sel], axis=1)
        else:
            part.on_joint = False
        part.resid = float(np.median(d))
        # the inlier half of the residuals IS the sensor noise on this surface
        inl = d[d <= max(np.median(d), 1e-4) * 2.0]
        sig = 1.4826 * float(np.median(np.abs(inl - np.median(inl)))) if inl.size \
            else cfg.sigma_floor
        part.sigma = float(np.clip(max(sig, np.median(inl) if inl.size else 0.0),
                                   cfg.sigma_floor, cfg.sigma_ceil))
        # a track only a few frames old has not earned a vote on articulation
        mature = self.track_born[sel] <= self.n - cfg.track_grace
        dm = d[mature] if mature.any() else d
        part.out_frac = float(np.mean(dm > cfg.split_out_band * part.sigma))
        part.n_mature = int(mature.sum())
        hot = (part.out_frac > cfg.split_out_frac) or (part.resid > cfg.split_res)
        part.over = part.over + 1 if hot else 0

    def _grow_co(self, n):
        """Keep the co-association matrices as wide as the track set."""
        m = self.co_same.shape[0]
        if n <= m:
            return
        for a in ("co_same", "co_seen"):
            M = getattr(self, a)
            N = np.zeros((n, n), np.float32)
            N[:m, :m] = M
            setattr(self, a, N)

    def _accumulate(self, part, cur, cur_ok, vis):
        """Soft votes for which points move together, over several motions."""
        cfg = self.cfg
        self._grow_co(len(self.anchor_xyz))
        idx = part.idx[(part.idx < len(cur_ok)) & (part.idx < len(self.anchor_ok))]
        # Excluding young tracks here as well was tried and measured worse --
        # pliers01 went from 2 parts to 5. Co-association needs the population;
        # the grace on the trigger and on the split is where it belongs.
        sel = idx[self.anchor_ok[idx] & cur_ok[idx] & vis[idx]]
        if sel.size < 2 * cfg.min_inliers or self.reg is None:
            return
        rem = np.ones(sel.size, bool)
        motions = []
        for _ in range(cfg.co_hyp):
            c = self.reg._RANSAC(p0=self.anchor_xyz[sel], tgt_pcd=cur[sel],
                                 w=None, remaining=rem, init_pose=None)
            if c is None:
                break
            motions.append(np.asarray(c["T"]))
        if len(motions) < 2:
            return
        D = np.stack([np.linalg.norm(
            self.anchor_xyz[sel] @ T[:3, :3].T + T[:3, 3] - cur[sel], axis=1)
            for T in motions])
        P = np.exp(-D / max(cfg.co_tau, 1e-6))
        P /= P.sum(axis=0, keepdims=True) + 1e-12
        # a frame where one motion explains everything says nothing about which
        # points move together, so it is down-weighted by how decisive it is
        w = float(np.mean(P.max(axis=0) > 0.8) * (1.0 - P.mean(axis=1).max()))
        if w <= 1e-3:
            return
        S = P.T @ P
        self.co_same[np.ix_(sel, sel)] += w * S
        self.co_seen[np.ix_(sel, sel)] += w

    def _fit_box(self, part):
        """Oriented box around this part's anchor-frame points."""
        try:
            from experiments.articulated.demo_part_discovery import fit_oriented_box
        except Exception:
            return None
        idx = part.idx[part.idx < len(self.anchor_xyz)]
        q = self.anchor_xyz[idx[self.anchor_ok[idx]]]
        if q.shape[0] < 6:
            return None
        try:
            # one stray track would otherwise stretch the box across the scene
            d = np.linalg.norm(q - np.median(q, axis=0), axis=1)
            return fit_oriented_box(q[d <= np.percentile(d, 90)])
        except Exception:
            return None

    def _reproject_owner(self, t2, depth):
        """Part id per pixel, from where each part's own points are right now.

        Not a rendering -- the tracked points ARE the surface we have observed,
        so the nearest one that also agrees in depth is the part that owns a
        pixel. A neighbouring part passing in front is near in the image and far
        in depth, and is rejected by the second test.
        """
        import cv2
        cfg = self.cfg
        cur, cok, cvi = self.cur
        H, W = depth.shape
        seed = np.zeros((H, W), np.uint8)
        lab = np.full((H, W), -1, np.int16)
        zbuf = np.zeros((H, W), np.float32)
        for j, p in enumerate(self.parts):
            idx = p.idx[(p.idx < len(cok)) & (p.idx < len(t2))]
            idx = idx[cok[idx] & cvi[idx]]
            if idx.size < 3:
                continue
            u = np.clip(t2[idx, 0].astype(int), 0, W - 1)
            v = np.clip(t2[idx, 1].astype(int), 0, H - 1)
            seed[v, u] = 1
            lab[v, u] = j
            zbuf[v, u] = cur[idx, 2]
        if not seed.any():
            return None, None
        dist, near = cv2.distanceTransformWithLabels(
            1 - seed, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.where(seed > 0)
        order = np.argsort(ys * W + xs)
        ys, xs = ys[order], xs[order]
        li = np.clip(near - 1, 0, len(ys) - 1)
        owner = lab[ys[li], xs[li]]
        zref = zbuf[ys[li], xs[li]]
        bad = (dist > cfg.reproj_px) | (np.abs(depth - zref) > cfg.reproj_depth) \
            | (depth <= 0)
        owner = np.where(bad, -1, owner)
        return owner, dist

    def _place_pending(self, i):
        """Give each waiting track to the part it actually moves with.

        For a track rigidly attached to part j, inv(T_j(t)) x(t) is the same
        point every frame. Its spread over time is therefore the evidence, and
        the winning part's mean IS the anchor -- computed from every frame the
        track was seen, not from the one it happened to be sampled in.
        """
        cfg = self.cfg
        cur, cok, cvi = self.cur
        placed, dropped = 0, 0
        for idx in list(self.pend):
            if idx >= len(cok):
                continue
            st = self.pend[idx]
            # Keep the raw observation and the poses that go with it. Throwing
            # the history away whenever the part list changed left every track
            # unowned right after a split, and the parts then lost the points
            # that justified it -- cabinet03 collapsed back to one body.
            if cok[idx] and cvi[idx]:
                st["obs"].append((cur[idx].copy(),
                                  [None if p.pose is None else p.pose.copy()
                                   for p in self.parts]))
                if len(st["obs"]) > cfg.pend_max_age:
                    st["obs"].pop(0)
            n_obs = len(st["obs"])
            if n_obs < cfg.pend_min_obs:
                if i - st["born"] > cfg.pend_max_age:
                    del self.pend[idx]
                    dropped += 1
                continue
            # re-evaluate against whatever the parts are NOW
            npart = len(self.parts)
            spread = np.full(npart, np.inf)
            mean = np.zeros((npart, 3))
            for j in range(npart):
                a = [(x - T[j][:3, 3]) @ T[j][:3, :3]
                     for x, T in st["obs"] if j < len(T) and T[j] is not None]
                if len(a) < cfg.pend_min_obs:
                    continue
                a = np.stack(a)
                mean[j] = a.mean(0)
                spread[j] = float(np.sqrt(a.var(0).sum()))
            order = np.argsort(spread)
            best = int(order[0])
            runner = spread[order[1]] if len(order) > 1 else np.inf
            noise = max(self.parts[best].sigma, cfg.sigma_floor)
            decisive = (spread[best] < cfg.pend_sigma * noise
                        and runner > cfg.pend_margin * spread[best])
            if not decisive:
                if i - st["born"] > cfg.pend_max_age:
                    del self.pend[idx]
                    dropped += 1
                continue
            # reprojection has to agree with the motion, or the track waits
            if cfg.reproject and self.reproj is not None:
                t2 = self.cur_tracks_2d
                if idx < len(t2):
                    u = int(np.clip(t2[idx, 0], 0, self.reproj.shape[1] - 1))
                    v = int(np.clip(t2[idx, 1], 0, self.reproj.shape[0] - 1))
                    o = int(self.reproj[v, u])
                    if o >= 0 and o != best:
                        if i - st["born"] > cfg.pend_max_age:
                            del self.pend[idx]
                            dropped += 1
                        continue
            self.anchor_xyz[idx] = mean[best].astype(np.float32)
            self.anchor_ok[idx] = True
            self.track_born[idx] = i        # the grace runs from placement
            p = self.parts[best]
            p.idx = np.unique(np.concatenate([p.idx, [idx]]))
            del self.pend[idx]
            placed += 1
        if placed or dropped:
            self.placed_total = getattr(self, "placed_total", 0) + placed
            self.dropped_total = getattr(self, "dropped_total", 0) + dropped
        return placed

    def _rot_floor(self, part):
        """Least rotational error a fit on this part's points can have."""
        idx = part.idx[part.idx < len(self.anchor_xyz)]
        q = self.anchor_xyz[idx[self.anchor_ok[idx]]]
        if q.shape[0] < 4:
            return 0.05
        gyr = float(np.sqrt(np.mean(np.sum((q - q.mean(0)) ** 2, axis=1))))
        return float(np.clip(part.sigma / max(gyr, 1e-3), 0.002, 0.5))

    def _joint_geom(self, j, part):
        """Parent and child points together, in the parent's frame at rest.

        Both sides matter: a cabinet door hinges on the body's edge, not inside
        the door, so judging the axis against the moving part alone would throw
        away every correct hinge.
        """
        par = self.parts[part.parent]
        def pts(q):
            idx = q.idx[q.idx < len(self.anchor_xyz)]
            idx = idx[self.anchor_ok[idx]]
            if idx.size > 300:
                idx = idx[np.linspace(0, idx.size - 1, 300).astype(int)]
            return self.anchor_xyz[idx]
        a, b = pts(par), pts(part)
        if a.shape[0] < 3 or b.shape[0] < 3:
            return None
        A0 = part.joint.A0
        if A0 is not None:
            b = b @ A0[:3, :3].T + A0[:3, 3]
        return np.concatenate([a, b]).astype(np.float64)

    def _merge_rigid(self, i):
        """Undo a split whose relative motion is best explained by no joint.

        A part built from a handful of points has a noisy pose, and that noise
        looks like relative motion -- 60 to 200 mm of it on RBO, far above the
        sensor noise. Nor can the residual decide: a 1-DoF model fits any jitter
        better than a rigid one, because random 3D points project onto their
        principal axis. What decides is that a real joint is DRIVEN, so its q(t)
        is smooth, while tracking error gives a white one.
        """
        cfg = self.cfg
        gone = []
        for j, p in enumerate(self.parts):
            jm, par = p.joint, p.parent
            # part 0's parent is 1 and part 1's parent is 0, so without this
            # both sides merge into each other and nothing is left
            if jm is None or par >= len(self.parts) or par == j:
                continue
            if j in gone or par in gone or len(self.parts) - len(gone) < 2:
                continue
            if len(jm.A) < cfg.merge_min_obs:
                continue
            c = jm.confidence()
            if c.get("smooth", 1.0) >= cfg.merge_smooth:
                continue        # q(t) is driven, so this is a joint
            keep = self.parts[par]
            keep.idx = np.unique(np.concatenate([keep.idx, p.idx]))
            keep.box = self._fit_box(keep)
            keep.born = self.n              # earn the right to split again
            gone.append(j)
            print(f"[naive] frame {i}: merged part {j} back into {par} "
                  f"(q(t) is not driven: smoothness {c['smooth']:.2f}, "
                  f"axis +-{c['axis_std_deg']:.1f} deg, {len(jm.A)} poses)")
        if not gone:
            return 0
        for j in sorted(gone, reverse=True):
            self.parts.pop(j)
        for k, p in enumerate(self.parts):
            p.parent = 0 if k else (1 if len(self.parts) > 1 else 0)
            if p.joint is None and k != p.parent:
                p.joint = JointModel()
        if len(self.parts) < 2:
            self.state = self.RIGID
        self.merged_total = getattr(self, "merged_total", 0) + len(gone)
        return len(gone)

    def _cohort_split(self, groups):
        """True when the two groups are just two different seeding batches."""
        tops = []
        for g in groups:
            g = np.asarray(g)
            b = self.track_born[g[g < len(self.track_born)]]
            if b.size < 3:
                return False
            vals, cnt = np.unique(b, return_counts=True)
            tops.append((vals[cnt.argmax()], cnt.max() / b.size))
        if len(tops) < 2:
            return False
        (b0, p0), (b1, p1) = tops[0], tops[1]
        return bool(b0 != b1 and min(p0, p1) >= self.cfg.cohort_pure)

    def _bic_split(self, part, sel, groups, motions, cur):
        """Is a second rigid body worth six more parameters?

        Same model selection as the joint type: n log-likelihood against
        k log n, with the noise the part's own fit reveals. A rigid part
        re-examined after more motion gains almost nothing and is left alone.
        """
        s2 = max(part.sigma, 1e-4) ** 2
        one = np.linalg.norm(
            self.anchor_xyz[sel] @ part.pose[:3, :3].T + part.pose[:3, 3]
            - cur[sel], axis=1)
        two = []
        for g, T in zip(groups, motions):
            g = np.asarray(g)
            if g.size == 0:
                continue
            two.append(np.linalg.norm(
                self.anchor_xyz[g] @ T[:3, :3].T + T[:3, 3] - cur[g], axis=1))
        if not two:
            return False, 0.0
        two = np.concatenate(two)
        n1, n2 = len(one), len(two)
        b1 = n1 * float(np.mean(one ** 2)) / s2 + 6 * np.log(max(n1, 2))
        b2 = n2 * float(np.mean(two ** 2)) / s2 + 12 * np.log(max(n2, 2))
        return bool(b2 + self.cfg.split_bic_margin < b1), float(b1 - b2)

    def _sep_sigma(self, groups, motions, part):
        """How far the groups' own points move if you swap the two motions.

        Degrees plus centimetres is scale-free nonsense; what decides whether a
        motion difference is real is how large the disagreement is where the
        points actually are, against the noise on those same points.
        """
        d = []
        for k, g in enumerate(groups):
            g = np.asarray(g)
            if g.size == 0:
                continue
            P0 = self.anchor_xyz[g]
            A = P0 @ motions[k][:3, :3].T + motions[k][:3, 3]
            B = P0 @ motions[1 - k][:3, :3].T + motions[1 - k][:3, 3]
            d.append(np.linalg.norm(A - B, axis=1))
        if not d:
            return 0.0
        return float(np.median(np.concatenate(d)) / max(part.sigma, 1e-6))

    def _cluster_co(self, sel):
        """Two groups from the accumulated affinity, or None if not separable."""
        cfg = self.cfg
        seen = self.co_seen[np.ix_(sel, sel)]
        if seen.size == 0 or np.median(seen) < cfg.co_min_seen:
            return None
        A = self.co_same[np.ix_(sel, sel)] / np.maximum(seen, 1e-6)
        keep = seen.sum(axis=1) > 0
        if keep.sum() < 2 * cfg.min_part_pts:
            return None
        A = A[np.ix_(keep, keep)]
        sub = sel[keep]
        try:
            from sklearn.cluster import AgglomerativeClustering
            lab = AgglomerativeClustering(
                n_clusters=2, metric="precomputed",
                linkage="average").fit_predict(1.0 - A)
        except Exception:
            return None
        g0, g1 = sub[lab == 0], sub[lab == 1]
        if min(len(g0), len(g1)) < cfg.min_part_pts:
            return None
        within = 0.5 * (A[np.ix_(lab == 0, lab == 0)].mean()
                        + A[np.ix_(lab == 1, lab == 1)].mean())
        across = A[np.ix_(lab == 0, lab == 1)].mean()
        if within - across < cfg.co_gap:
            return None
        return [g0, g1], float(within - across)

    def _split(self, j, cur, cur_ok, vis):
        """Second rigid motion inside one part; ambiguous points are dropped."""
        cfg = self.cfg
        part = self.parts[j]
        idx = part.idx[(part.idx < len(cur_ok)) & (part.idx < len(self.anchor_ok))]
        idx = idx[self.track_born[idx] <= self.n - cfg.track_grace]
        sel = idx[self.anchor_ok[idx] & cur_ok[idx] & vis[idx]]
        if sel.size < 2 * cfg.min_part_pts:
            return False

        # Stage B first: groups that held together over frames, not one frame's
        # RANSAC. The single frame decides only when the history is too thin.
        cc = self._cluster_co(sel) if cfg.persist else None
        if cc is not None:
            groups, gap = cc
            motions = []
            for g in groups:
                c = self.reg._RANSAC(p0=self.anchor_xyz[g], tgt_pcd=cur[g],
                                     w=None, remaining=np.ones(len(g), bool),
                                     init_pose=None)
                motions.append(np.asarray(c["T"]) if c is not None else part.pose)
            ss = self._sep_sigma(groups, motions, part)
            if ss >= cfg.split_sep_sigma:
                if cfg.cohort_veto and self._cohort_split(groups):
                    self.cohort_blocked = getattr(self, "cohort_blocked", 0) + 1
                    return False
                ok, dbic = ((True, float("nan")) if not cfg.split_bic
                            else self._bic_split(part, sel, groups, motions, cur))
                if ok:
                    return self._accept(j, part, groups, motions, cur, cur_ok,
                                        "coassoc", gap, None, ss, dbic)
                self.bic_blocked = getattr(self, "bic_blocked", 0) + 1

        if cfg.persist and (not cfg.frame_fallback
                            or part.over < cfg.frame_after * cfg.split_frames):
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
        ss = self._sep_sigma([sel, sel], motions, part)
        if ss < cfg.split_sep_sigma:
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
        # splitting must explain the points better than the one body did
        before = float(np.median(np.linalg.norm(
            self.anchor_xyz[sel] @ part.pose[:3, :3].T + part.pose[:3, 3]
            - cur[sel], axis=1)))
        after = float(np.median(np.concatenate(
            [D[k][keep & (best == k)] for k in range(2)])))
        if before > 1e-6 and after > cfg.split_gain * before:
            return False
        if cfg.cohort_veto and self._cohort_split(groups):
            self.cohort_blocked = getattr(self, "cohort_blocked", 0) + 1
            return False
        dbic = float("nan")
        if cfg.split_bic:
            ok, dbic = self._bic_split(part, sel, groups, motions, cur)
            if not ok:
                self.bic_blocked = getattr(self, "bic_blocked", 0) + 1
                return False
        return self._accept(j, part, groups, motions, cur, cur_ok, "frame",
                            float("nan"), keep, ss, dbic)

    def _accept(self, j, part, groups, motions, cur, cur_ok, via, gap, keep,
                sep_sigma=float("nan"), dbic=float("nan")):
        """Install the two groups as parts and hand the gaussian labels over."""
        order = np.argsort([-len(g) for g in groups])
        new = [NaivePart(idx=np.asarray(groups[k]), pose=motions[k],
                         born=self.n, view_dirs=[np.array([0., 0., 1.])])
               for k in order]
        # the affinity that justified this split has been spent; each new part
        # has to earn its own before it may split again
        all_idx = np.concatenate([np.asarray(g) for g in groups])
        self.co_same[np.ix_(all_idx, all_idx)] = 0.0
        self.co_seen[np.ix_(all_idx, all_idx)] = 0.0
        self.parts[j:j + 1] = new
        for k, p in enumerate(self.parts):
            p.parent = 0 if k else (1 if len(self.parts) > 1 else 0)
            if p.joint is None and k != p.parent:
                p.joint = JointModel()
        if self.model is not None:
            self._split_labels(j, new, cur, cur_ok)
        for p in self.parts:
            p.box = self._fit_box(p)
        self.state = self.SPLIT
        self.split_log.append((self.n, len(self.parts)))
        # Points seeded together share one anchor pose, so a pose error at that
        # moment is written into all of them as one rigid offset -- which looks
        # exactly like a second part. If a group is one birth cohort, that is
        # what happened, not articulation.
        coh = []
        for g in groups:
            g = np.asarray(g)
            b = self.track_born[g[g < len(self.track_born)]]
            if b.size:
                vals, cnt = np.unique(b, return_counts=True)
                coh.append(f"{100 * cnt.max() / b.size:.0f}%@f{vals[cnt.argmax()]}")
        self.last_cohort = coh

        # how much relative motion it actually took, which is the number the
        # "wiggle and it becomes controllable" claim lives or dies on
        mm = sep_sigma * part.sigma * 1000
        self.split_evidence = (self.n, mm, part.sigma * 1000)
        print(f"[naive] split at frame {self.n} via {via}: {len(self.parts)} "
              f"parts (motion {mm:.0f} mm = {sep_sigma:.1f} sigma, "
              f"noise {part.sigma * 1000:.1f} mm, outliers "
              f"{100 * part.out_frac:.0f}%, "
              f"{len(groups[0])}/{len(groups[1])} pts"
              + (f", gap {gap:.2f}" if gap == gap else "")
              + (f", dBIC {dbic:.0f}" if dbic == dbic else "")
              + (f", cohorts {'/'.join(coh)}" if coh else "")
              + (f", {int((~keep).sum())} dropped" if keep is not None else "")
              + ")")
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
        """Tracks coloured by part, an oriented 3D box and pose axes per part,
        and the fitted joint axis -- the same read-out as the dense pipeline."""
        import cv2
        from point2pose.utils.visualization import draw_oriented_3d_box
        pal = palette or [(60, 140, 235), (200, 120, 40), (70, 180, 90),
                          (200, 80, 200), (60, 200, 200), (90, 90, 235)]
        K = self.K
        out = bgr
        cur, cur_ok, vis = getattr(self, "cur", (None, None, None))
        t2 = getattr(self, "cur_tracks_2d", None)
        if t2 is None:
            return out

        m = getattr(self, "mask", None)
        if m is not None:
            edge = cv2.morphologyEx((m > 0).astype(np.uint8), cv2.MORPH_GRADIENT,
                                    np.ones((3, 3), np.uint8))
            out[edge > 0] = (235, 235, 235)
        owner = np.full(len(t2), -1)
        for j, p in enumerate(self.parts):
            owner[p.idx[p.idx < len(t2)]] = j
        for k, (x, y) in enumerate(t2):
            if not (cur_ok[k] and vis[k]):
                continue
            col = (110, 110, 110) if owner[k] < 0 else pal[owner[k] % len(pal)]
            cv2.circle(out, (int(x), int(y)), 3, col, -1, cv2.LINE_AA)

        def proj(P):
            if P[2] <= 1e-3:
                return None
            return (int(P[0] * K[0, 0] / P[2] + K[0, 2]),
                    int(P[1] * K[1, 1] / P[2] + K[1, 2]))

        for j, p in enumerate(self.parts):
            col = pal[j % len(pal)]
            if p.box is None:
                p.box = self._fit_box(p)
            if p.box is not None and p.pose is not None:
                try:
                    out = draw_oriented_3d_box(K, out, p.pose, p.box,
                                               line_color=col, linewidth=2)
                except Exception:
                    pass
            idx = p.idx[p.idx < len(self.anchor_xyz)]
            q = self.anchor_xyz[idx[self.anchor_ok[idx]]]
            if q.shape[0] < 3 or p.pose is None:
                continue
            c = q.mean(0)
            org = p.pose[:3, :3] @ c + p.pose[:3, 3]
            p0 = proj(org)
            for ax, acol in zip(np.eye(3) * 0.05,
                                [(60, 60, 235), (60, 220, 60), (235, 160, 60)]):
                p1 = proj(p.pose[:3, :3] @ (c + ax) + p.pose[:3, 3])
                if p0 and p1:
                    cv2.arrowedLine(out, p0, p1, acol, 2, cv2.LINE_AA,
                                    tipLength=0.3)
            lab = f"p{j} {p.resid * 1000:.0f}mm {100 * p.out_frac:.0f}%"
            if p.joint is not None and p.joint.kind:
                c = p.joint.confidence()
                # the type on its own says nothing; how sure and how much it has
                # been moved are what tell a person to keep wiggling
                lab += (f" {p.joint.kind[:4]} {c['conf']:.2f}"
                        f"/t{c['type_p']:.2f}/x{c['excitation']:.1f}"
                        + ("*" if p.on_joint else ""))
            if p0:
                cv2.putText(out, lab, (p0[0] - 30, p0[1] - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

        # the joint axis, drawn once per pair in the parent's frame
        drawn = set()
        for j, p in enumerate(self.parts):
            jm = p.joint
            if jm is None or jm.kind is None or p.parent >= len(self.parts):
                continue
            key = tuple(sorted((j, p.parent)))
            if key in drawn:
                continue
            drawn.add(key)
            Tp = self.parts[p.parent].pose
            if Tp is None:
                continue
            base = jm.point if (jm.kind == "revolute" and jm.point is not None) \
                else np.zeros(3)
            a0 = Tp[:3, :3] @ (base - jm.axis * 0.08) + Tp[:3, 3]
            a1 = Tp[:3, :3] @ (base + jm.axis * 0.08) + Tp[:3, 3]
            q0, q1 = proj(a0), proj(a1)
            if q0 and q1:
                cv2.line(out, q0, q1, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(out, jm.kind, q1, cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (20, 20, 20), 1, cv2.LINE_AA)
        return out

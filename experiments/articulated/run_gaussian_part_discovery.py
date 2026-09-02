"""Part discovery with sparse hypotheses and dense rendering-based assignment.

Sparse RANSAC proposes rigid-motion hypotheses; Gaussians lifted from depth
decide, per pixel, which hypothesis actually explains each surface. Each half
does what it is good at: RANSAC is an efficient hypothesis generator even from
few tracks, while the dense residual does not care how many keypoints landed on
a part -- which is what defeated every sparse-only detector on thin and
symmetric objects.

Hypotheses come from the same sequential RANSAC as Point2Pose's registration
(Lin et al., ECCV 2026); the dense assignment follows the plan's section 3, with
log-odds accumulated over time rather than a per-frame decision.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.data_types.frame import Frame
from point2pose.io.sources.dataset.sapien_reader import open_sequence
from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)
from point2pose.modules.tracker.tapir_tracker import TapirTracker
from point2pose.pipeline.components.gaussian_part_assignment import (
    GaussianCloud, GaussianPartAssignment,
)
from scipy.spatial.transform import Rotation as _R

PAL = np.array([[245, 135, 66], [60, 130, 214], [100, 190, 90],
                [220, 100, 170], [230, 200, 80]], np.uint8)


from point2pose.pipeline.components.joint_model import JointModel


def _se3_pow(T, s):
    """T raised to a real power, along its own screw axis."""
    from scipy.linalg import logm, expm
    if abs(s - 1.0) < 1e-9:
        return np.asarray(T)
    try:
        return np.real(expm(np.real(logm(np.asarray(T, dtype=np.float64))) * s))
    except Exception:
        return np.asarray(T)


def sample_points(mask, depth, n, border=4):
    m = cv2.erode(mask.astype(np.uint8), np.ones((border, border), np.uint8))
    ys, xs = np.where((m > 0) & (depth > 0.05))
    if len(xs) == 0:
        raise SystemExit("empty init mask")
    k = np.arange(len(xs)) if len(xs) <= n else \
        np.random.default_rng(0).choice(len(xs), n, replace=False)
    return np.stack([xs[k], ys[k]], 1).astype(np.float32)


def lift(pts, depth, K):
    H, W = depth.shape
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, H - 1)
    z = depth[y, x]
    p = np.zeros((len(pts), 3))
    p[:, 0] = (pts[:, 0] - K[0, 2]) * z / K[0, 0]
    p[:, 1] = (pts[:, 1] - K[1, 2]) * z / K[1, 1]
    p[:, 2] = z
    return p, z > 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--out", default="debug/multi-parts/runs/gauss/out.mp4")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--gauss-stride", type=int, default=0,
                    help="depth subsampling stride for the gaussian cloud. "
                         "0 picks it automatically from --gauss-target.")
    ap.add_argument("--gauss-target", type=int, default=3000,
                    help="gaussians the automatic stride aims for")
    ap.add_argument("--inlier-thres", type=float, default=0.008)
    ap.add_argument("--max-hyp", type=int, default=4)
    ap.add_argument("--hyp-mode", choices=["baselines", "window"], default="window",
                    help="window reproduces the hypothesis generation that scored "
                         "best on RBO: a FIXED-length trailing window, using only "
                         "tracks observed at both ends, shrinking the window when "
                         "too few qualify. baselines is the earlier fixed-lag scheme.")
    ap.add_argument("--win-steps", type=int, default=20,
                    help="trailing window length, in processed frames")
    ap.add_argument("--win-min-span", type=int, default=4)
    ap.add_argument("--auto-baselines", type=int, default=0,
                    help="pick this many baselines per frame by measuring which "
                         "interval a single rigid motion explains WORST. That is "
                         "exactly where the parts disagree, and it adapts to an "
                         "object that opens and closes, where a fixed anchor sees "
                         "almost no net displacement.")
    ap.add_argument("--auto-lag-max", type=int, default=90)
    ap.add_argument("--baselines", default="0",
                    help="comma-separated frame lags to generate hypotheses from, "
                         "in addition to the anchor. A part that opens and closes "
                         "has little NET displacement from a fixed anchor -- on "
                         "RBO one hypothesis matched all three parts to within "
                         "25 mm -- so hypotheses must also come from recent, "
                         "shorter intervals where the relative motion is large.")
    ap.add_argument("--min-inliers", type=int, default=10)
    ap.add_argument("--depth-sigma", type=float, default=0.02)
    ap.add_argument("--color-weight", type=float, default=0.3)
    ap.add_argument("--object-masks", default=None,
                    help="npz from run_sam2_propagate.py. On RBO the GT masks are "
                         "amodal and paint over the hand; SAM2 masks cut it out, so "
                         "they are the right object mask for real data.")
    ap.add_argument("--mode", choices=["render", "pointwise"], default="pointwise",
                    help="pointwise asks where each gaussian goes under each "
                         "hypothesis; render rasterises each hypothesis in full")
    ap.add_argument("--coassoc", action="store_true", default=True,
                    help="cluster a co-association matrix instead of carrying "
                         "hypothesis slots across frames")
    ap.add_argument("--no-coassoc", dest="coassoc", action="store_false")
    ap.add_argument("--coverage-weight", type=float, default=0.3,
                    help="unused; coverage now breaks ties rather than trading "
                         "against the split ratio (see combined())")
    ap.add_argument("--merge-rigid", type=int, default=1,
                    help="also offer a grouping in which groups that never move "
                         "relative to each other are merged")
    ap.add_argument("--score-round", type=int, default=3,
                    help="decimals the split ratio is rounded to before coverage "
                         "and part count break the tie. At 2 a harmful merge ties "
                         "with the grouping it damages and wins on fewer parts, "
                         "costing RBO cabinet02 a drawer.")
    ap.add_argument("--merge-eps", type=float, default=0.002,
                    help="how much split ratio a merged grouping may give up and "
                         "still be preferred. The window is narrow and measured: "
                         "collapsing the simulated pliers from 5 groups to 2 costs "
                         "0.001, while the merge that absorbs a real RBO drawer "
                         "(cabinet02, 3/3 at 85.9%% -> 2/3 at 73.4%%) costs 0.004. "
                         "At 0.03 the harmful merge wins.")
    ap.add_argument("--merge-tol", type=float, default=0.012,
                    help="median point displacement below which two groups are "
                         "treated as one rigid body, metres")
    ap.add_argument("--winsets", type=int, default=1,
                    help="also group by clustering the recurring decisive winner "
                         "sets, and let the GT-free score choose")
    ap.add_argument("--winset-min-members", type=int, default=3)
    ap.add_argument("--winset-thresh", type=float, default=0.5)
    ap.add_argument("--winset-max-frac", type=float, default=0.5,
                    help="largest winner set kept, as a fraction of the cloud")
    ap.add_argument("--winset-min-frac", type=float, default=0.02,
                    help="smallest winner set kept, as a fraction of the cloud")
    ap.add_argument("--coassoc-weight", choices=["none", "decisive", "disagree"],
                    default="none",
                    help="weight each frame's co-association evidence by how far "
                         "its posterior is from a single hypothesis owning "
                         "everything")
    ap.add_argument("--co-sample", type=int, default=4000)
    ap.add_argument("--em-iters", type=int, default=1,
                    help="M-step rounds per frame: refine each hypothesis against "
                         "the dense model, then re-assign")
    ap.add_argument("--grow-every", type=int, default=0,
                    help="add gaussians for unexplained object pixels every N "
                         "frames (0 = never). Without it the model is whatever "
                         "the first frame saw.")
    ap.add_argument("--grow-tol", type=float, default=0.03)
    ap.add_argument("--co-max-k", type=int, default=5)
    ap.add_argument("--co-frac", type=float, default=0.5,
                    help="fraction of co-decided frames a pair must share a group in")
    ap.add_argument("--min-group", type=int, default=30,
                    help="drop groups smaller than this; a handful of gaussians is "
                         "not a part, it is a stray assignment")
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--track-grow", type=int, default=2,
                    help="grow each part's dense model every N tracked frames")
    ap.add_argument("--track-grow-tol", type=float, default=0.02)
    ap.add_argument("--track-grow-max", type=int, default=1500)
    ap.add_argument("--track-color", type=float, default=0.0,
                    help="weight of the colour term in the tracking energy. Off: "
                         "measured, it makes the laptop lid 25x worse (1125 mm vs "
                         "44 mm). The lid's colours are its screen face, and once "
                         "the back turns to the camera every colour disagrees, so "
                         "the energy starts preferring poses that hide the part.")
    ap.add_argument("--dense-hyp", type=int, default=0,
                    help="propose an extra motion hypothesis by fitting the "
                         "surface no existing hypothesis explains")
    ap.add_argument("--dense-hyp-tol", type=float, default=0.02,
                    help="depth residual above which a gaussian counts as "
                         "unexplained, metres")
    ap.add_argument("--dense-hyp-min", type=int, default=200)
    ap.add_argument("--max-step", type=float, default=0.2,
                    help="largest translation a part may move between two tracked "
                         "frames, metres; 0 disables the guard")
    ap.add_argument("--exclusive-mask", type=int, default=1)
    ap.add_argument("--joint-tol", type=float, default=0.02,
                    help="how far off the joint manifold a pose may be and still "
                         "be used to refine the joint, metres")
    ap.add_argument("--track-grow-gate", type=float, default=0.015,
                    help="only grow a part whose tracking energy is below this")
    ap.add_argument("--track-rmax", type=float, default=0.05,
                    help="truncation of the per-part tracking energy, metres")
    args = ap.parse_args()

    r = open_sequence(args.seq_dir)
    parts = r.get_object_names()

    ext = None
    if args.object_masks:
        d = np.load(args.object_masks, allow_pickle=True)
        ext = {int(f): (d["masks"][k].max(axis=0) > 0).astype(np.uint8)
               for k, f in enumerate(d["frames"]) if k < len(d["masks"])}
        print(f"[g] using SAM2 object masks for {len(ext)} frames")

    def object_mask(i):
        if ext is not None:
            return ext.get(i)
        m = np.zeros((r.H, r.W), np.uint8)
        for x in r.get_masks(i):
            m |= x
        return m
    K, H, W = r.K, r.H, r.W
    frames = list(range(0, len(r), args.stride))
    a = frames[0]

    if ext is not None:
        frames = [i for i in frames if ext.get(i) is not None]
        a = frames[0]
    rgb0, d0 = r.get_color(a), r.get_depth(a)
    obj0 = object_mask(a)
    if obj0 is None or obj0.sum() < 200:
        raise SystemExit("no usable object mask on the anchor frame")

    # Choose the sampling stride from how many gaussians the object actually
    # yields, not from a fixed number. A stride tuned on a 17k-gaussian RBO
    # cabinet leaves a pair of pliers with 648: too sparse for anything to be
    # decisively assigned, and the grouping then labels 14% of the object.
    # Measured: dropping the stride to 1 on the five RBO pliers sequences took
    # coverage from 1.4/2 to 1.8/2. The stride is a density knob, so set it by
    # density.
    if args.gauss_stride <= 0:
        gs = 4
        while gs > 1:
            n_est = int((obj0[::gs, ::gs] > 0).sum())
            if n_est >= args.gauss_target:
                break
            gs -= 1
        args.gauss_stride = gs
        print(f"[g] gaussian stride {gs} (target {args.gauss_target}, "
              f"object is {int((obj0 > 0).sum())} px)")
    cloud = GaussianCloud.from_depth(rgb0, d0, K, obj0, stride=args.gauss_stride)
    assign = GaussianPartAssignment(cloud, args.max_hyp,
                                    depth_sigma=args.depth_sigma,
                                    color_weight=args.color_weight)
    assign.coassoc_weight = args.coassoc_weight
    assign.winset_max_frac = args.winset_max_frac
    if args.coassoc:
        assign.init_coassoc(args.co_sample)
    print(f"[g] {len(cloud)} gaussians, GT parts {parts}, {len(frames)} frames")

    pim = r.render_part_index_map(a)[::args.gauss_stride, ::args.gauss_stride]
    dd = d0[::args.gauss_stride, ::args.gauss_stride]
    mm = obj0[::args.gauss_stride, ::args.gauss_stride]
    gt_lab = pim[(dd > 0.05) & (mm > 0)]

    tracker = TapirTracker({"checkpoint_path": args.checkpoint,
                            "resize_height": 480, "resize_width": 480,
                            "visible_threshold": 0.5, "device": "cuda"})
    pts0 = sample_points(obj0, d0, args.n_points)
    f0 = Frame(id=a, rgb=rgb0, depth=d0, intrinsics=K)
    tracker.add_query_points(f0, pts0)
    tracker.initialize(f0)
    anchor_xyz, anchor_ok = lift(pts0, d0, K)

    reg = SVDClusterRANSACRegister({
        "ransac_iters": 200, "sample_size": 4,
        "inlier_thres": args.inlier_thres, "min_inliers": args.min_inliers,
        "max_clusters": args.max_hyp, "use_uncertainty": False})

    lags = [int(x) for x in args.baselines.split(",")]
    order, anchor_to = [], {}

    def rigid_residual(P, Q):
        """How bimodal are the residuals of a single rigid fit?

        Not the residual magnitude: that grows with tracking drift, so ranking on
        it simply picked the longest interval available every time (lag 90 chosen
        36 times out of 145). The ratio of a high percentile to the median is
        scale-free -- one noisy rigid body gives ~2-3, two parts moving apart give
        far more, and common drift raises both ends together.
        """
        if len(P) < 6:
            return -1.0
        cp, cq = P.mean(0), Q.mean(0)
        H = (P - cp).T @ (Q - cq)
        try:
            U, _, Vt = np.linalg.svd(H)
        except np.linalg.LinAlgError:
            return -1.0
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        e = np.sqrt((((Q - cq) - (P - cp) @ R.T) ** 2).sum(1))
        med = float(np.median(e))
        if med < 1e-6:
            return -1.0
        return float(np.percentile(e, 90) / med)

    def pick_lags(i, cur, cur_ok, vis):
        """Rank candidate intervals by how badly one rigid motion explains them."""
        cands = []
        for j in sorted(past.keys()):
            lag = i - j
            if lag <= 0 or lag > args.auto_lag_max:
                continue
            sp, so, sv = past[j]
            u = np.where(so & sv & cur_ok & vis)[0]
            if len(u) < 3 * args.min_inliers:
                continue
            cands.append((rigid_residual(sp[u], cur[u]), lag))
        cands.sort(reverse=True)
        return [0] + [l for _, l in cands[: args.auto_baselines]]
    past, rel_to_anchor = {}, {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer, hist = None, []
    track_hist = {}        # frame -> (xyz, ok, visible) of the sparse tracks
    pose_log = []          # per-frame refined transform of every hypothesis
    T_anchor_gt = {p: r.get_gt_pose(a, p) for p in parts}

    for n, i in enumerate(frames):
        rgb, dep = r.get_color(i), r.get_depth(i)
        tracks, _, vis = tracker.track_once(Frame(id=i, rgb=rgb, depth=dep, intrinsics=K))
        vis = vis.astype(bool)
        cur, cur_ok = lift(tracks, dep, K)
        use = np.where(anchor_ok & cur_ok & vis)[0]

        past[i] = (cur.copy(), cur_ok.copy(), vis.copy())
        track_hist[int(i)] = (cur.copy(), cur_ok.copy(), vis.copy())
        order.append(i)

        # ---- anchor->current dominant motion, maintained every frame so any
        # windowed hypothesis can be composed into the anchor frame the gaussians
        # live in. Leaving this to the windowed pass itself was the bug that made
        # the "trailing window only" experiment collapse to identity.
        u0 = np.where(anchor_ok & cur_ok & vis)[0]
        if len(u0) >= args.min_inliers:
            c0 = reg._RANSAC(p0=anchor_xyz[u0], tgt_pcd=cur[u0], w=None,
                             remaining=np.ones(len(u0), bool), init_pose=None)
            if c0 is not None:
                anchor_to[i] = c0["T"]
        if i not in anchor_to:
            anchor_to[i] = anchor_to.get(order[-2], np.eye(4)) if len(order) > 1 else np.eye(4)

        hyps = []
        if args.hyp_mode == "window":
            # tracks must be observed at BOTH ends of one fixed interval, or the
            # fit compares different time spans against each other
            for span in (args.win_steps, args.win_steps // 2, args.win_steps // 4):
                if span < args.win_min_span or len(order) <= span:
                    continue
                j = order[-1 - span]
                sp, so, sv = past[j]
                u = np.where(so & sv & cur_ok & vis)[0]
                if len(u) < 2 * args.min_inliers:
                    continue
                rem = np.ones(len(u), bool)
                got = []
                for _ in range(args.max_hyp):
                    c = reg._RANSAC(p0=sp[u], tgt_pcd=cur[u], w=None,
                                    remaining=rem, init_pose=None)
                    if c is None:
                        break
                    got.append(c["T"] @ anchor_to[j])
                if got:
                    hyps.extend(got)
                    break
            # always keep the anchor-to-current motion as one hypothesis
            hyps.append(anchor_to[i])
            n_real = len(hyps)
            uniq = []
            for T in hyps:
                if all(np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.008 or
                       np.linalg.norm(T[:3, :3] - U[:3, :3]) > 0.02 for U in uniq):
                    uniq.append(T)
            hyps = uniq[:args.max_hyp]

        frame_lags = pick_lags(i, cur, cur_ok, vis) if args.auto_baselines > 0 else lags
        if args.hyp_mode == "window":
            frame_lags = []
        # pool hypotheses over several time baselines, then keep the distinct ones
        for lag in frame_lags:
            if lag == 0:
                srcp, srco = anchor_xyz, anchor_ok
            else:
                j = i - lag
                if j not in past:
                    continue
                srcp, srco, srcv = past[j]
                srco = srco & srcv
            u2 = np.where(srco & cur_ok & vis)[0]
            if len(u2) < args.min_inliers:
                continue
            remaining = np.ones(len(u2), bool)
            for _ in range(args.max_hyp):
                c = reg._RANSAC(p0=srcp[u2], tgt_pcd=cur[u2], w=None,
                                remaining=remaining, init_pose=None)
                if c is None:
                    break
                # express every hypothesis anchor->current so they are comparable
                T = c["T"] if lag == 0 else c["T"] @ rel_to_anchor.get(j, np.eye(4))
                hyps.append(T)
        # drop near-duplicates so the log-odds table is not wasted on copies
        uniq = []
        for T in hyps:
            if all(np.linalg.norm(T[:3, 3] - U[:3, 3]) > 0.01 or
                   np.linalg.norm(T[:3, :3] - U[:3, :3]) > 0.03 for U in uniq):
                uniq.append(T)
        hyps = uniq[:args.max_hyp]
        if hyps:
            rel_to_anchor[i] = hyps[0]
        n_real = len(hyps)
        if not hyps:
            hyps = [np.eye(4)]; n_real = 0
        # pad so the log-odds table keeps a fixed width
        while len(hyps) < args.max_hyp:
            hyps.append(hyps[-1])

        om = object_mask(i)
        if om is None:
            continue
        if args.mode == "pointwise":
            hy = list(hyps[:args.max_hyp])
            # EM: assign softly, then refine each hypothesis against the surface
            # it is supposed to explain, then assign again with the refined ones
            for _ in range(max(0, args.em_iters)):
                Rm = assign.residuals(hy, K, H, W, dep, obs_mask=om)
                P = assign.soft_membership(Rm)
                hy = [assign.refine_pose(T, P[k], K, H, W, dep, obs_mask=om)
                      for k, T in enumerate(hy)]
            # ---- dense hypothesis generation ----
            # On real data the bottleneck is not the assignment but the
            # hypotheses: measured on RBO, one sparse RANSAC hypothesis matches
            # all three drawers to within 6-25 mm, because they open AND close,
            # so net displacement from a fixed anchor stays comparable to both
            # the depth noise and the inlier threshold. Sparse keypoints also
            # have to LAND on a part before it can be proposed at all. The
            # surface that no hypothesis explains is itself the evidence for a
            # motion nobody has proposed yet, so fit one to it directly.
            if args.dense_hyp:
                import torch as _t
                Rm = assign.residuals(hy, K, H, W, dep, obs_mask=om)
                bestr = Rm.min(dim=0).values
                un = (bestr > args.dense_hyp_tol) & (bestr < 1.0)
                if int(un.sum()) >= args.dense_hyp_min:
                    w_un = un.float().detach().cpu().numpy()
                    cand = []
                    for T0 in hy:
                        Tr = assign.refine_pose(T0, w_un, K, H, W, dep, iters=10,
                                                huber=0.04, obs_mask=om)
                        cand.append((assign.fit_energy(Tr, w_un, K, H, W, dep,
                                                       obs_mask=om), Tr))
                    e_n, T_n = min(cand, key=lambda x: x[0])
                    novel = all(
                        np.linalg.norm(T_n[:3, 3] - U[:3, 3]) > 0.01 or
                        np.linalg.norm(T_n[:3, :3] - U[:3, :3]) > 0.03 for U in hy)
                    st_dense = 1 if (novel and e_n < args.dense_hyp_tol) else 0
                    if st_dense:
                        hy = hy[:max(1, args.max_hyp - 1)] + [T_n]
                    st_diag = (int(un.sum()), float(e_n), int(novel))
                else:
                    st_dense = 0
                    st_diag = (int(un.sum()), -1.0, -1)
            st = assign.step_pointwise(hy, K, H, W, dep, obs_mask=om)
            if args.dense_hyp:
                st["dense_hyp"] = st_dense
                st["dense_diag"] = st_diag
            hyps = hy + hyps[len(hy):]
            # the refined transforms ARE the per-part 6-DoF poses; keep them so
            # they can be scored against ground truth once groups are known
            # keep each refined hypothesis together with WHICH gaussians it
            # explains, so a part can later pick its own pose per frame instead
            # of trusting a slot index whose meaning changes every frame
            Pm = assign.soft_membership(
                assign.residuals(hy, K, H, W, dep, obs_mask=om))
            pose_log.append({
                "frame": int(i),
                "T": [np.asarray(T) for T in hy],
                "P": Pm.detach().cpu().numpy(),
            })
            if args.grow_every > 0 and n % args.grow_every == 0 and n > 0:
                added = assign.grow(rgb, dep, K, om, hy, hy[0],
                                    tol=args.grow_tol, stride=args.gauss_stride)
                st["grown"] = int(added)
        else:
            st = assign.step(hyps[:args.max_hyp], K, H, W, dep, rgb, obs_mask=om)
        st["n_hypotheses"] = int(n_real)
        st["lags"] = [int(x) for x in frame_lags]
        st["n_uniq"] = int(len(hyps))
        # what GT part does each hypothesis's winning set actually consist of?
        comp = []
        for w in getattr(assign, "last_wins", []):
            g = gt_lab[w] if w.shape[0] == gt_lab.shape[0] else np.array([])
            g = g[g >= 0]
            if g.size < 10:
                comp.append(None); continue
            c = np.bincount(g, minlength=len(parts))
            comp.append({"dom": parts[int(np.argmax(c))],
                         "pur": round(float(c.max()/c.sum()), 3), "n": int(g.size)})
        st["win_gt"] = comp
        st["hyp_spread_mm"] = float(1000 * max(
            [np.linalg.norm(np.asarray(a)[:3,3] - np.asarray(b)[:3,3])
             for a in hyps for b in hyps] or [0.0]))
        st["n_usable_tracks"] = int(len(use))
        hist.append(st)

        lab = assign.labels()
        vis_img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        # draw each gaussian where its OWN assigned hypothesis puts it this frame,
        # so the overlay tracks the parts instead of sitting at anchor pixels
        means = cloud.means.detach().cpu().numpy()
        for h in range(args.max_hyp):
            sel_h = lab == h
            if not np.any(sel_h):
                continue
            T = np.asarray(hyps[h])
            q = means[sel_h] @ T[:3, :3].T + T[:3, 3]
            good = q[:, 2] > 1e-3
            u = (q[good, 0] * K[0, 0] / q[good, 2] + K[0, 2])
            v = (q[good, 1] * K[1, 1] / q[good, 2] + K[1, 2])
            c = tuple(int(x) for x in PAL[h % len(PAL)])
            for uu, vv in zip(u.astype(int), v.astype(int)):
                if 0 <= uu < W and 0 <= vv < H:
                    cv2.circle(vis_img, (uu, vv), 1, c, -1)
        und = lab < 0
        if np.any(und):
            T = np.asarray(hyps[0])
            q = means[und] @ T[:3, :3].T + T[:3, 3]
            good = q[:, 2] > 1e-3
            u = (q[good, 0] * K[0, 0] / q[good, 2] + K[0, 2]).astype(int)
            v = (q[good, 1] * K[1, 1] / q[good, 2] + K[1, 2]).astype(int)
            for uu, vv in zip(u, v):
                if 0 <= uu < W and 0 <= vv < H:
                    cv2.circle(vis_img, (uu, vv), 1, (110, 110, 110), -1)
        bar = np.full((30, W, 3), (28, 24, 20), np.uint8)
        cv2.putText(bar, f"frame {i:3d}  hypotheses {len(set(map(id, hyps)))}  "
                         f"explained {100*st['explained_frac']:.0f}%  "
                         f"decided {100*(lab>=0).mean():.0f}%",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([vis_img, bar])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                     (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

    if writer:
        writer.release()

    def grouping_score(labels_full):
        """How well does a grouping explain the per-frame posteriors?

        Ground-truth-free model selection. For each frame and each group, take the
        hypothesis that best explains that group and read off how much posterior
        mass it actually captures. A grouping that cuts along the real parts
        collects nearly all of it; one that splits a rigid body does not.
        """
        if not pose_log:
            return 0.0, 0.0
        gs = [g for g in set(labels_full.tolist()) if g >= 0]
        if len(gs) < 1:
            return 0.0, 0.0
        # Score the split against NOT splitting. Asking only "how well does each
        # group explain the posteriors" rewards a single group, which is why the
        # laptop collapsed to one part as soon as depth noise was added. The
        # honest question is whether splitting explains the data better than
        # treating everything as one body.
        tot, cnt = 0.0, 0
        for rec in pose_log[::3]:
            P = rec["P"]
            if P.shape[1] != labels_full.shape[0]:
                continue
            union = labels_full >= 0
            if union.sum() < 20:
                continue
            base = float(P[:, union].sum(axis=1).max())
            split = 0.0
            for g in gs:
                m = labels_full == g
                if m.sum() < 10:
                    continue
                split += float(P[:, m].sum(axis=1).max())
            if base > 1e-9:
                tot += split / base
                cnt += 1
        # Scaled by how much of the object the grouping actually labels. The
        # ratio alone is relative to whatever subset a candidate chose, so a
        # grouping that labels only the easy third of the gaussians scores a
        # smaller, easier problem and wins -- that is how winner-set grouping
        # came to over-segment the simulated pliers 5 ways. Making the
        # denominator global instead swings the other way and rewards ONE group
        # covering everything, which lost the scissors blade entirely (1/2 at
        # 56%). Relative split times coverage keeps both honest.
        cov = float((labels_full >= 0).mean())
        return tot / max(cnt, 1), cov

    # both groupings are computed, then the better-scoring one is kept: the hard
    # log-odds path wins on scissors and pliers, co-association on laptop and
    # eyeglasses, and neither dominates.
    cand = []
    lab_hard = assign.labels()
    cand.append(("labels", lab_hard, np.arange(len(cloud))) + (grouping_score(lab_hard),))
    if args.coassoc:
        lab_co, sub_co = assign.coassoc_labels(args.co_frac, args.min_group,
                                               max_k=args.co_max_k)
        full = np.full(len(cloud), -1, dtype=int)
        full[sub_co] = lab_co
        cand.append(("coassoc", lab_co, sub_co) + (grouping_score(full),))
    if args.winsets:
        _ws = getattr(assign, "winsets", [])
        if _ws:
            _sz = np.array([len(x) for x in _ws])
            print(f"[g] winner sets recorded: {len(_ws)} over {len(hist)} frames "
                  f"({len(_ws)/max(len(hist),1):.2f}/frame), size median "
                  f"{int(np.median(_sz))} of {len(cloud)} "
                  f"({100*np.median(_sz)/len(cloud):.1f}%)")
        else:
            print(f"[g] winner sets recorded: NONE over {len(hist)} frames")
        lab_ws, sub_ws = assign.winset_labels(min_members=args.winset_min_members,
                                              thresh=args.winset_thresh,
                                              min_frac=args.winset_min_frac)
        if (lab_ws >= 0).any():
            cand.append(("winsets", lab_ws, sub_ws) + (grouping_score(lab_ws),))
    def merge_rigid_groups(labels_full, tol=0.012):
        """Merge groups that never move relative to each other.

        Clustering recurring winner sets recovers all three RBO drawers but cuts
        the object into more pieces than there are parts (6 groups for 3 on
        cabinet01) -- a surface can win under two hypotheses that mean the same
        motion. Two groups are the same rigid body exactly when the transforms
        that best explain them move their points identically, which is a
        ground-truth-free test on quantities already computed.
        """
        gs = sorted(g for g in set(labels_full.tolist()) if g >= 0)
        if len(gs) < 2 or not pose_log:
            return labels_full
        gmeans = cloud.means.detach().cpu().numpy()
        pts = {}
        for g in gs:
            m = np.where(labels_full == g)[0]
            if m.size > 400:
                m = m[np.linspace(0, m.size - 1, 400).astype(int)]
            pts[g] = gmeans[m]
        seq = {g: [] for g in gs}
        for rec in pose_log[::2]:
            P, Ts = rec["P"], rec["T"]
            if P.shape[1] != labels_full.shape[0]:
                continue
            for g in gs:
                m = labels_full == g
                if m.sum() < 10:
                    seq[g].append(None)
                    continue
                seq[g].append(np.asarray(Ts[int(np.argmax(P[:, m].sum(axis=1)))]))
        parent = {g: g for g in gs}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, a_ in enumerate(gs):
            for b_ in gs[i + 1:]:
                X = np.concatenate([pts[a_], pts[b_]], 0)
                d = []
                for Ta, Tb in zip(seq[a_], seq[b_]):
                    if Ta is None or Tb is None:
                        continue
                    pa = X @ Ta[:3, :3].T + Ta[:3, 3]
                    pb = X @ Tb[:3, :3].T + Tb[:3, 3]
                    d.append(float(np.linalg.norm(pa - pb, axis=1).mean()))
                # a high percentile, not the median: two groups are one body
                # only if they NEVER move apart. On RBO the drawers agree with
                # the body in most frames -- exactly the pathology that makes
                # the sequence hard -- and a median test merged all six groups
                # into one.
                if d and float(np.percentile(d, 90)) < tol:
                    ra, rb = find(a_), find(b_)
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
        if nxt < len(gs):
            print(f"[g] merged {len(gs)} groups into {nxt} "
                  f"(groups moving identically are one body)")
        return out

    if args.merge_rigid:
        merged = []
        for nm, lb, sb, sc in cand:
            full = np.full(len(cloud), -1, dtype=int)
            full[sb] = lb
            mf = merge_rigid_groups(full, tol=args.merge_tol)
            if (mf >= 0).any() and len(set(mf[mf >= 0].tolist())) < \
                    len(set(full[full >= 0].tolist())):
                merged.append((nm + "+merge", mf[sb], sb, grouping_score(mf)))
        cand.extend(merged)
    # ties go to the grouping with fewer parts: merging two groups that pick the
    # same hypothesis leaves `split` unchanged, so a correct merge scores exactly
    # the same as the over-segmented version it replaces
    def combined(e):
        # Coverage BREAKS TIES; it does not trade against the split ratio. Added
        # with a weight it silently overrides the ratio: on the simulated
        # eyeglasses a 2-group labelling at 87% coverage (ratio 1.015) beat the
        # correct 3-group one at 73% (ratio 1.036) and lost a temple. Rounding
        # the ratio first says plainly what is meant -- when two groupings
        # explain the posteriors about equally well, prefer the one that labels
        # more of the object, and then the one with fewer parts.
        r, c = e[3]
        return round(r, args.score_round), c
    cand.sort(key=lambda x: (-combined(x)[0], -combined(x)[1],
                             len(set(x[1][x[1] >= 0].tolist()))))
    print("[g] grouping selection: " +
          "  ".join(f"{n}={sc[0]:.3f}x{sc[1]:.2f}"
                    f"[{len(set(lb[lb >= 0].tolist()))}]"
                    for n, lb, _, sc in cand))
    # Prefer the merged variant of whatever wins unless the score clearly
    # disagrees. The merge test is itself evidence -- it only fires when two
    # groups never move apart -- and the score has no complexity term, so an
    # over-segmented grouping edges out its own merge by a hair. A per-group
    # penalty is not the fix: any penalty large enough to matter here also makes
    # a single group beat the correct 2-group split on the simulated scissors.
    best_name = cand[0][0]
    if not best_name.endswith("+merge"):
        for e in cand[1:]:
            if e[0] == best_name + "+merge" and \
                    combined(e)[0] >= combined(cand[0])[0] - args.merge_eps:
                print("[g] taking the merged variant (same score, fewer parts)")
                cand = [e] + [c for c in cand if c is not e]
                break
    chosen, lab, sub, _ = cand[0]
    print(f"[g] using '{chosen}' grouping")
    gt_lab_full = gt_lab
    gt_lab = gt_lab[sub]

    if False:
        # does the accumulated matrix separate same-part from cross-part pairs?
        import torch as _t
        sub_np = assign.co_idx.cpu().numpy()
        A = (assign.co_same / assign.co_seen.clamp(min=1.0)).cpu().numpy()
        g = gt_lab[sub_np]
        same = (g[:, None] == g[None, :]) & (g[:, None] >= 0)
        cross = (g[:, None] != g[None, :]) & (g[:, None] >= 0) & (g[None, :] >= 0)
        iu = np.triu_indices(len(g), 1)
        sv, cv = A[iu][same[iu]], A[iu][cross[iu]]
        seen = assign.co_seen.cpu().numpy()[iu]
        print(f"[g] co-association: same-part pairs mean {sv.mean():.3f} (n={sv.size}), "
              f"cross-part mean {cv.mean():.3f} (n={cv.size}), gap {sv.mean()-cv.mean():+.3f}")
        print(f"[g] co_seen per pair: mean {seen.mean():.1f} of {len(frames)} frames, "
              f"zero for {100*(seen==0).mean():.1f}% of pairs")
    ok = lab >= 0
    print(f"\n[g] decided {100*ok.mean():.1f}% of gaussians")
    used = sorted(set(lab[ok].tolist()))
    covered = set()
    rows = []
    for h in used:
        m = lab == h
        if int(m.sum()) < args.min_group:
            continue
        g = gt_lab[m]
        g = g[g >= 0]
        if g.size == 0:
            continue
        cnt = np.bincount(g, minlength=len(parts))
        dom = int(np.argmax(cnt))
        covered.add(parts[dom])
        pur = float(cnt[dom] / cnt.sum())
        rows.append({"hypothesis": int(h), "n": int(m.sum()),
                     "dominant_gt": parts[dom], "purity": pur})
        print(f"   hyp {h}: {int(m.sum()):6d} gaussians  dominant {parts[dom]:8s} "
              f"purity {100*pur:5.1f}%")
    print(f"[g] GT parts covered {len(covered)}/{len(parts)} {sorted(covered)}   "
          f"groups {len(rows)}")

    # ---- per-part 6-DoF tracking pass ----
    # Reading a pose out of hypothesis slot h is meaningless: the slot means a
    # different motion every frame, which is why the pose error was excellent at
    # the median and catastrophic at p90. Once the groups are known, each part is
    # tracked in its own right -- its membership is fixed, and its pose is
    # refined frame to frame, warm-started from the previous estimate.
    # sparse tracks are needed again for per-part hypotheses, so keep them
    part_tracks, part_members, part_joints = {}, {}, {}
    if rows:
        import torch as _T
        # Build every part's fixed membership first, then walk the sequence once
        # with all parts together. Tracking them independently let two parts
        # settle on the same surface: the laptop lid, segmented correctly at 94%,
        # registered itself onto the base and came out 800 mm away. Parts are
        # mutually exclusive in the image, so each one is refined against the
        # object mask minus what the others already occupy.
        infos = []
        for row in rows:
            h = row["hypothesis"]
            if args.coassoc:
                w = np.zeros(len(cloud), dtype=np.float32)
                w[sub[lab == h]] = 1.0
            else:
                w = (lab == h).astype(np.float32)
            if w.sum() < 20:
                continue
            # which sparse tracks sit on this part? nearest-gaussian lookup in
            # the anchor frame, so the part gets its own hypothesis generator
            part_track_idx = None
            gm = cloud.means.detach().cpu().numpy()
            mem = gm[w > 0.5]
            if mem.shape[0] >= 10 and anchor_xyz.shape[0] > 0:
                from scipy.spatial import cKDTree
                tree = cKDTree(mem)
                dist, _ = tree.query(anchor_xyz, k=1)
                thr = max(0.01, float(np.percentile(dist, 20)) * 2.0)
                part_track_idx = np.where(dist < thr)[0]
            infos.append({"row": row, "w": w, "tidx": part_track_idx,
                          "T_prev": None, "track": []})
        # Parts are ordered biggest first and each is jointed to the biggest
        # OTHER part. Making the largest part the sole parent is wrong: on the
        # laptop the lid carries more gaussians than the base, so it became the
        # reference and the part that actually needed constraining had no joint
        # at all. A joint constrains a pair, and either end can be the reference.
        infos.sort(key=lambda d: -float(d["w"].sum()))
        for j, inf in enumerate(infos):
            inf["parent"] = (1 if j == 0 else 0) if len(infos) > 1 else j
            inf["joint"] = JointModel() if len(infos) > 1 else None
            inf["jv"] = []

        nrec, n_grown = 0, 0
        for rec in pose_log:
            dep_i = r.get_depth(rec["frame"])
            om_i = object_mask(rec["frame"])
            if om_i is None:
                continue
            om_b = om_i > 0
            rgb_i = r.get_color(rec["frame"]) if args.track_color > 0 else None
            occ = [assign.occupancy(inf["T_prev"], inf["w"], K, H, W, dilate=2)
                   if inf["T_prev"] is not None else None for inf in infos]
            Pm = rec["P"]
            for j, inf in enumerate(infos):
                w = inf["w"]
                # the object mask minus the pixels the OTHER parts hold, keeping
                # anything this part itself already covers
                others = None
                for k2, o in enumerate(occ):
                    if k2 == j or o is None:
                        continue
                    others = o if others is None else (others | o)
                om_j = om_i
                if args.exclusive_mask and others is not None:
                    if occ[j] is not None:
                        others = others & (~occ[j])
                    cand_m = om_b & (~others)
                    if cand_m.sum() >= 200:
                        om_j = cand_m.astype(np.uint8)

                score = (Pm * w[None, :Pm.shape[1]]).sum(axis=1)
                k = int(np.argmax(score))
                T_prev = inf["T_prev"]
                T0 = rec["T"][k] if score[k] > 0 else (
                    T_prev if T_prev is not None else np.eye(4))
                # Refine from BOTH the previous pose and the hypothesis, and keep
                # whichever ends up explaining the surface better. Incremental
                # tracking alone diverged late in the sequence (331 mm on the last
                # third of scissors) once the part had moved far from its anchor;
                # the hypothesis alone is coarse. Either can rescue the other.
                cands = [T0]
                if T_prev is not None:
                    cands.append(T_prev)
                    # constant velocity. A part swinging fast (the laptop lid
                    # covers 130 deg) leaves the projective-ICP basin in one
                    # step, and its own sparse tracks are gone by then because
                    # the face they sit on has rotated away from the camera.
                    dT = inf.get("dT")
                    if dT is not None:
                        # fractional powers of the last increment: the lid turns
                        # at a near-constant rate, so the right pose lies on the
                        # screw through dT, but not always at exactly one step
                        for sfac in (0.5, 1.0, 1.5, 2.0):
                            cands.append(_se3_pow(dT, sfac) @ T_prev)
                # every hypothesis, not only the best-scoring one: once a part's
                # tracks are lost its own membership score is meaningless, but
                # another group's hypothesis may still carry the right motion
                for Th in rec["T"]:
                    cands.append(np.asarray(Th))
                # ---- the joint turns a 6-DoF search into a scalar one ----
                jm = inf["joint"]
                Tp = infos[inf["parent"]]["T_prev"] if jm is not None else None
                if jm is not None and jm.kind is not None and Tp is not None:
                    vs_ = inf["jv"]
                    span = max(max(vs_) - min(vs_), 1e-3)
                    pred = vs_[-1] + (vs_[-1] - vs_[-2] if len(vs_) > 1 else 0.0)
                    # A local sweep around the predicted value AND a scan of the
                    # joint's whole physical range. Deriving the scan range from
                    # the motion seen so far cannot work: the laptop lid is only
                    # 15 degrees into a 130-degree swing when its joint is first
                    # fitted, so a range-relative grid never reaches where the
                    # part actually is once tracking slips.
                    full = (np.linspace(-np.pi, np.pi, 121)
                            if jm.kind == "revolute"
                            else np.linspace(-0.6, 0.6, 121))
                    grid = np.concatenate([
                        pred + np.linspace(-1.0, 1.0, 41) * max(span, 0.1), full])
                    Tg = [Tp @ jm.at(float(v)) for v in grid]
                    eg = assign.fit_energy_batch(Tg, w, K, H, W, dep_i,
                                                 obs_mask=om_j,
                                                 r_max=args.track_rmax,
                                                 obs_rgb=rgb_i,
                                                 color_w=args.track_color)
                    for oi in np.argsort(eg)[:3]:
                        cands.append(Tg[int(oi)])
                # A hypothesis fitted to THIS part's own tracks. The global
                # hypotheses are dominated by the biggest surface, so a small
                # thin part undergoing a large rotation never gets a good
                # initialisation from them -- which is exactly where tracking
                # still diverged (scissors blade, eyeglasses temple).
                tidx = inf["tidx"]
                if tidx is not None and tidx.size >= 6:
                    th = track_hist.get(int(rec["frame"]))
                    if th is not None:
                        cu, cok, cvi = th
                        sel_t = tidx[anchor_ok[tidx] & cok[tidx] & cvi[tidx]]
                        if sel_t.size >= 6:
                            rem = np.ones(sel_t.size, bool)
                            c = reg._RANSAC(p0=anchor_xyz[sel_t],
                                            tgt_pcd=cu[sel_t], w=None,
                                            remaining=rem, init_pose=None)
                            if c is not None:
                                cands.append(c["T"])
                best_T, best_e, best_i = None, float("inf"), -1
                dbg = []
                # A part cannot teleport between two consecutive frames. Without
                # this the joint sweep -- which deliberately proposes poses all
                # over the joint's range so a lost part can be recovered -- also
                # lets a part that was tracking perfectly jump: the storage
                # cabinet's static body, tracked to 0.7 mm for two thirds of the
                # sequence, ended up 815 mm away.
                for ci, Tc in enumerate(cands):
                    if T_prev is not None and args.max_step > 0:
                        d = float(np.linalg.norm(np.asarray(Tc)[:3, 3] - T_prev[:3, 3]))
                        if d > args.max_step:
                            continue
                    Tr = assign.refine_pose(Tc, w, K, H, W, dep_i, iters=8,
                                            huber=0.04, obs_mask=om_j)
                    # truncated energy over ALL the part's gaussians, not a median
                    # over the survivors: otherwise the candidate that abandons the
                    # part scores best (see fit_energy).
                    e = assign.fit_energy(Tr, w, K, H, W, dep_i, obs_mask=om_j,
                                          r_max=args.track_rmax, obs_rgb=rgb_i,
                                          color_w=args.track_color)
                    dbg.append((ci, e, float(np.linalg.norm(Tr[:3, 3]))))
                    if e < best_e:
                        best_T, best_e, best_i = Tr, e, ci
                if best_T is not None:
                    # coarse-to-fine: the wide huber above buys the basin, a
                    # tight one buys the accuracy
                    Tf = assign.refine_pose(best_T, w, K, H, W, dep_i, iters=8,
                                            huber=0.006, obs_mask=om_j)
                    ef = assign.fit_energy(Tf, w, K, H, W, dep_i, obs_mask=om_j,
                                           r_max=args.track_rmax, obs_rgb=rgb_i,
                                           color_w=args.track_color)
                    if ef < best_e:
                        best_T, best_e = Tf, ef
                T = best_T if best_T is not None else np.eye(4)
                inf["e"] = best_e
                jm = inf["joint"]
                Tp = infos[inf["parent"]]["T_prev"] if jm is not None else None
                if jm is not None and Tp is not None and best_e < args.track_grow_gate:
                    A = np.linalg.inv(Tp) @ T
                    jm.add(A)
                    # refit as evidence accumulates: the type and the axis both
                    # sharpen once the part has actually swung
                    if jm.kind is None or len(jm.A) % 4 == 0:
                        was = jm.kind
                        if jm.fit():
                            inf["jv"] = [jm.value_of(a_) for a_ in jm.A]
                            _ = was
                    elif jm.kind is not None:
                        inf["jv"].append(jm.value_of(A))
                if T_prev is not None:
                    inf["dT"] = T @ np.linalg.inv(T_prev)
                if os.environ.get("TRACK_DEBUG") and eval(os.environ.get("TRACK_DEBUG_F", "rec['frame'] % 10 == 0")):
                    try:
                        gtp_ = inf["row"]["dominant_gt"]
                        Tg = r.get_gt_pose(rec["frame"], gtp_) @ np.linalg.inv(
                            T_anchor_gt[gtp_])
                        eg = assign.fit_energy(Tg, w, K, H, W, dep_i,
                                               obs_mask=om_j, r_max=args.track_rmax)
                        egf = assign.fit_energy(Tg, w, K, H, W, dep_i,
                                                obs_mask=om_i, r_max=args.track_rmax)
                        dbg.append(("gt", eg, float(np.linalg.norm(Tg[:3, 3]))))
                        dbg.append(("gtfull", egf, 0.0))
                    except Exception as ex:
                        pass
                    print(f"   [dbg] f{rec['frame']:3d} part{j} pick {best_i} "
                          + "  ".join(f"c{c}:e={e:.4f},|t|={t:.3f}"
                                      for c, e, t in dbg))
                inf["T_prev"] = T
                inf["track"].append((int(rec["frame"]), T.copy()))

            # ---- online part modelling ----
            # Every part now has a pose for this frame, so any object pixel none
            # of them explains is newly revealed surface. Give it to the part it
            # borders and anchor it through that part's pose: the model of each
            # part fills in as the object articulates, which is what lets a face
            # that was hidden at the anchor frame ever be tracked.
            if args.track_grow > 0 and nrec % args.track_grow == 0:
                poses = [inf["T_prev"] for inf in infos]
                ws = [inf["w"] for inf in infos]
                ok_g = [inf.get("e", 1e9) < args.track_grow_gate for inf in infos]
                n_new, owner = assign.grow_parts(
                    r.get_color(rec["frame"]), dep_i, K, om_i, poses, ws,
                    tol=args.track_grow_tol, stride=args.gauss_stride,
                    max_new=args.track_grow_max, grow_ok=ok_g)
                if n_new:
                    for j, inf in enumerate(infos):
                        add = (owner == j).astype(np.float32)
                        inf["w"] = np.concatenate([inf["w"], add])
                    n_grown += n_new
            nrec += 1

        for j, inf in enumerate(infos):
            jm = inf["joint"]
            if jm is None:
                continue
            gt_ = inf["row"]["dominant_gt"]
            if jm.kind is None:
                print(f"[g] joint {gt_}: not fitted ({len(jm.A)} trusted frames)")
            else:
                rng = (np.degrees(max(inf["jv"]) - min(inf["jv"]))
                       if jm.kind == "revolute" else
                       1000 * (max(inf["jv"]) - min(inf["jv"])))
                print(f"[g] joint {gt_} -> "
                      f"{infos[inf['parent']]['row']['dominant_gt']}"
                      f": {jm.kind} axis {np.round(jm.axis, 3)}  range "
                      f"{rng:.0f}{'deg' if jm.kind == 'revolute' else 'mm'}"
                      f"  from {len(jm.A)} frames")
        if args.track_grow > 0:
            print(f"[g] part modelling grew the cloud by {n_grown} gaussians "
                  f"({len(cloud)} total)")
        for inf in infos:
            gtp_ = inf["row"]["dominant_gt"]
            part_tracks[gtp_] = inf["track"]
            # the grown membership and the fitted joint, for the demo
            part_members[gtp_] = inf["w"] > 0.5
            par_ = infos[inf["parent"]]["row"]["dominant_gt"] if inf["joint"] else gtp_
            part_joints[gtp_] = (inf["joint"], par_)

        print("[g] per-part 6-DoF tracking (fixed membership, warm-started):")
        for gtp, track in part_tracks.items():
            te, re_ = [], []
            for (i, T) in track:
                Tgt = r.get_gt_pose(i, gtp) @ np.linalg.inv(T_anchor_gt[gtp])
                E = np.linalg.inv(Tgt) @ T
                te.append(float(np.linalg.norm(E[:3, 3])))
                re_.append(float(np.degrees(
                    np.linalg.norm(_R.from_matrix(E[:3, :3]).as_rotvec()))))
            te, re_ = np.array(te), np.array(re_)
            fr = np.array([i for (i, _) in track])
            # where in the sequence does it break? A part cannot be tracked before
            # it has moved enough to be distinguishable, so the early frames are
            # reported separately rather than folded into one number.
            n3 = max(1, len(te) // 3)
            seg = [(f"first {n3}", te[:n3], re_[:n3]),
                   (f"middle", te[n3:2*n3], re_[n3:2*n3]),
                   (f"last {len(te)-2*n3}", te[2*n3:], re_[2*n3:])]
            print(f"   {gtp:8s}: median {1000*np.median(te):6.1f} mm {np.median(re_):5.2f} deg"
                  f"   p90 {1000*np.percentile(te,90):6.1f} mm {np.percentile(re_,90):5.2f} deg")
            for nm_, t_, r_ in seg:
                if t_.size:
                    print(f"       {nm_:10s} median {1000*np.median(t_):6.1f} mm "
                          f"{np.median(r_):5.2f} deg   worst {1000*t_.max():6.1f} mm")
            for row in rows:
                if row["dominant_gt"] == gtp:
                    row["track_trans_mm"] = float(np.median(te) * 1000)
                    row["track_rot_deg"] = float(np.median(re_))

    # ---- pose read straight off the hypothesis slots, for comparison ----
    if pose_log and rows and False:
        print("[g] per-part pose error against ground truth:")
        for row in rows:
            h = row["hypothesis"]
            gtp = row["dominant_gt"]
            te, re_ = [], []
            for rec in pose_log:
                if h >= len(rec["T"]):
                    continue
                Test = np.asarray(rec["T"][h])
                Tgt = r.get_gt_pose(rec["frame"], gtp) @ np.linalg.inv(T_anchor_gt[gtp])
                E = np.linalg.inv(Tgt) @ Test
                te.append(float(np.linalg.norm(E[:3, 3])))
                re_.append(float(np.degrees(
                    np.linalg.norm(_R.from_matrix(E[:3, :3]).as_rotvec()))))
            if te:
                te, re_ = np.array(te), np.array(re_)
                row["trans_err_mm"] = float(np.median(te) * 1000)
                row["rot_err_deg"] = float(np.median(re_))
                print(f"   {gtp:8s} (hyp {h}): median {1000*np.median(te):6.1f} mm  "
                      f"{np.median(re_):5.2f} deg   "
                      f"p90 {1000*np.percentile(te,90):6.1f} mm "
                      f"{np.percentile(re_,90):5.2f} deg")
    if rows:
        print(f"[g] mean purity {100*np.mean([x['purity'] for x in rows]):.1f}%")
    # ---- demo video: dense part model + oriented box + pose axes per part ----
    if part_tracks:
        from experiments.articulated.demo_part_discovery import fit_oriented_box
        from point2pose.utils.visualization import draw_oriented_3d_box
        gm = cloud.means.detach().cpu().numpy()
        boxes, members = {}, {}
        for row in rows:
            h = row["hypothesis"]; gtp = row["dominant_gt"]
            if args.coassoc:
                w = np.zeros(len(cloud), bool); w[sub[lab == h]] = True
            else:
                w = (lab == h)
            # prefer the membership as it stands after online growth: that is
            # the model the tracker actually used, and showing the frozen
            # first-frame one hides the part modelling entirely
            w = part_members.get(gtp, w)
            if w.shape[0] < gm.shape[0]:
                w = np.concatenate([w, np.zeros(gm.shape[0] - w.shape[0], bool)])
            members[gtp] = w
            try:
                # trim the outer few percent before fitting the box: online
                # growth occasionally attaches a stray pixel through a slipped
                # pose, and one such point inflates an oriented box visibly
                q = gm[w]
                d = np.linalg.norm(q - np.median(q, axis=0), axis=1)
                boxes[gtp] = fit_oriented_box(q[d <= np.percentile(d, 97)])
            except Exception:
                boxes[gtp] = None
        pose_at = {g: dict(t) for g, t in part_tracks.items()}

        dw = None
        demo_path = os.path.splitext(args.out)[0] + "_demo.mp4"
        for i in frames:
            im = cv2.cvtColor(r.get_color(i), cv2.COLOR_RGB2BGR).copy()
            for k, (gtp, w) in enumerate(members.items()):
                T = pose_at.get(gtp, {}).get(int(i))
                if T is None:
                    continue
                col = tuple(int(x) for x in PAL[k % len(PAL)])
                q = gm[w] @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
                ok = q[:, 2] > 1e-3
                u = (q[ok, 0] * K[0, 0] / q[ok, 2] + K[0, 2]).astype(int)
                v = (q[ok, 1] * K[1, 1] / q[ok, 2] + K[1, 2]).astype(int)
                for uu, vv in zip(u, v):
                    if 0 <= uu < W and 0 <= vv < H:
                        cv2.circle(im, (uu, vv), 1, col, -1)
                if boxes.get(gtp) is not None:
                    try:
                        im = draw_oriented_3d_box(K, im, np.asarray(T), boxes[gtp],
                                                  line_color=col, linewidth=2)
                    except Exception:
                        pass
                # pose axes at the part centroid
                c = gm[w].mean(0)
                org = np.asarray(T)[:3, :3] @ c + np.asarray(T)[:3, 3]
                for ax, acol in zip(np.eye(3) * 0.05,
                                    [(60, 60, 235), (60, 220, 60), (235, 160, 60)]):
                    tip = np.asarray(T)[:3, :3] @ (c + ax) + np.asarray(T)[:3, 3]
                    if org[2] > 1e-3 and tip[2] > 1e-3:
                        p0 = (int(org[0]*K[0,0]/org[2]+K[0,2]), int(org[1]*K[1,1]/org[2]+K[1,2]))
                        p1 = (int(tip[0]*K[0,0]/tip[2]+K[0,2]), int(tip[1]*K[1,1]/tip[2]+K[1,2]))
                        cv2.arrowedLine(im, p0, p1, acol, 2, cv2.LINE_AA, tipLength=0.3)
            # the estimated joint axis, drawn through the parent's pose. Both
            # ends of a pair carry a model of the same joint, so draw it once.
            drawn = set()
            for k, (gtp, w) in enumerate(members.items()):
                jm, par = part_joints.get(gtp, (None, None))
                if jm is None or jm.kind is None or par == gtp:
                    continue
                key = frozenset((gtp, par))
                if key in drawn:
                    continue
                drawn.add(key)
                Tp = pose_at.get(par, {}).get(int(i))
                if Tp is None:
                    continue
                Tp = np.asarray(Tp)
                # the fitted point is only determined up to a slide along the
                # axis -- (I-R) annihilates the axis direction -- so anchor the
                # drawn segment at the point nearest the part it belongs to
                c = gm[w].mean(0)
                base = (jm.point + jm.axis * float((c - jm.point) @ jm.axis)
                        if jm.kind == "revolute" else c)
                seg = np.stack([base - jm.axis * 0.12, base + jm.axis * 0.12])
                q = seg @ Tp[:3, :3].T + Tp[:3, 3]
                if (q[:, 2] > 1e-3).all():
                    pu = (q[:, 0] * K[0, 0] / q[:, 2] + K[0, 2]).astype(int)
                    pv = (q[:, 1] * K[1, 1] / q[:, 2] + K[1, 2]).astype(int)
                    cv2.line(im, (pu[0], pv[0]), (pu[1], pv[1]), (255, 255, 255),
                             4, cv2.LINE_AA)
                    cv2.line(im, (pu[0], pv[0]), (pu[1], pv[1]), (40, 40, 40),
                             2, cv2.LINE_AA)
                    cv2.putText(im, jm.kind, (pu[1] + 6, pv[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1,
                                cv2.LINE_AA)
            bar = np.full((30, W, 3), (26, 22, 18), np.uint8)
            cv2.putText(bar, f"frame {i:4d}   {len(members)} parts tracked   "
                             f"dense model + 6-DoF pose + joint axis",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1,
                        cv2.LINE_AA)
            canvas = np.vstack([im, bar])
            if dw is None:
                dw = cv2.VideoWriter(demo_path, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                     (canvas.shape[1], canvas.shape[0]))
            dw.write(canvas)
        if dw:
            dw.release()
            print(f"[g] wrote {demo_path}")

    json.dump({"sequence": os.path.basename(args.seq_dir), "gt_parts": parts,
               "groups": rows, "covered": sorted(covered),
               "history": hist},
              open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=2)
    print(f"[g] wrote {args.out}")


if __name__ == "__main__":
    main()

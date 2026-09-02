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
    ap.add_argument("--gauss-stride", type=int, default=2)
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

    cloud = GaussianCloud.from_depth(rgb0, d0, K, obj0, stride=args.gauss_stride)
    assign = GaussianPartAssignment(cloud, args.max_hyp,
                                    depth_sigma=args.depth_sigma,
                                    color_weight=args.color_weight)
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
            st = assign.step_pointwise(hy, K, H, W, dep, obs_mask=om)
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
            return -1.0
        gs = [g for g in set(labels_full.tolist()) if g >= 0]
        if len(gs) < 1:
            return -1.0
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
            m_all = P[:, union].sum(axis=1)
            base = float(m_all.max())
            split = 0.0
            for g in gs:
                m = labels_full == g
                if m.sum() < 10:
                    continue
                split += float(P[:, m].sum(axis=1).max())
            if base > 1e-9:
                tot += split / base
                cnt += 1
        return tot / max(cnt, 1)

    # both groupings are computed, then the better-scoring one is kept: the hard
    # log-odds path wins on scissors and pliers, co-association on laptop and
    # eyeglasses, and neither dominates.
    cand = []
    lab_hard = assign.labels()
    cand.append(("labels", lab_hard, np.arange(len(cloud)), grouping_score(lab_hard)))
    if args.coassoc:
        lab_co, sub_co = assign.coassoc_labels(args.co_frac, args.min_group,
                                               max_k=args.co_max_k)
        full = np.full(len(cloud), -1, dtype=int)
        full[sub_co] = lab_co
        cand.append(("coassoc", lab_co, sub_co, grouping_score(full)))
    cand.sort(key=lambda x: -x[3])
    print("[g] grouping selection: " +
          "  ".join(f"{n}={sc:.3f}" for n, _, _, sc in cand))
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
    part_tracks = {}
    if rows:
        import torch as _T
        lab_full = lab if not args.coassoc else None
        for row in rows:
            h = row["hypothesis"]
            if args.coassoc:
                w = np.zeros(len(cloud), dtype=np.float32)
                w[sub[lab == h]] = 1.0
            else:
                w = (lab == h).astype(np.float32)
            if w.sum() < 20:
                continue
            # For each frame take the hypothesis that best explains THIS group --
            # that is the coarse initialisation the sparse side is good at -- then
            # refine it densely. Tracking from identity instead failed outright on
            # the moving parts (330 mm error) because projective association from
            # a stale pose lands on the wrong surface.
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

            track = []
            T_prev = None
            for rec in pose_log:
                Pm = rec["P"]
                if Pm.shape[1] != w.shape[0]:
                    continue
                score = (Pm * w[None, :]).sum(axis=1)
                k = int(np.argmax(score))
                T0 = rec["T"][k] if score[k] > 0 else (
                    T_prev if T_prev is not None else np.eye(4))
                dep_i = r.get_depth(rec["frame"])
                om_i = object_mask(rec["frame"])
                if om_i is None:
                    continue
                # Refine from BOTH the previous pose and the hypothesis, and keep
                # whichever ends up explaining the surface better. Incremental
                # tracking alone diverged late in the sequence (331 mm on the last
                # third of scissors) once the part had moved far from its anchor;
                # the hypothesis alone is coarse. Either can rescue the other.
                cands = [T0]
                if T_prev is not None:
                    cands.append(T_prev)
                # A hypothesis fitted to THIS part's own tracks. The global
                # hypotheses are dominated by the biggest surface, so a small
                # thin part undergoing a large rotation never gets a good
                # initialisation from them -- which is exactly where tracking
                # still diverged (scissors blade, eyeglasses temple).
                if part_track_idx is not None and part_track_idx.size >= 6:
                    th = track_hist.get(int(rec["frame"]))
                    if th is not None:
                        cu, cok, cvi = th
                        sel_t = part_track_idx[
                            anchor_ok[part_track_idx] & cok[part_track_idx]
                            & cvi[part_track_idx]]
                        if sel_t.size >= 6:
                            rem = np.ones(sel_t.size, bool)
                            c = reg._RANSAC(p0=anchor_xyz[sel_t],
                                            tgt_pcd=cu[sel_t], w=None,
                                            remaining=rem, init_pose=None)
                            if c is not None:
                                cands.append(c["T"])
                best_T, best_e = None, float("inf")
                for Tc in cands:
                    Tr = assign.refine_pose(Tc, w, K, H, W, dep_i, iters=8,
                                            obs_mask=om_i)
                    e = assign.fit_error(Tr, w, K, H, W, dep_i, obs_mask=om_i)
                    if e < best_e:
                        best_T, best_e = Tr, e
                T = best_T if best_T is not None else np.eye(4)
                T_prev = T
                track.append((int(rec["frame"]), T.copy()))
            part_tracks[row["dominant_gt"]] = track

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
            members[gtp] = w
            try:
                boxes[gtp] = fit_oriented_box(gm[w])
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
            bar = np.full((30, W, 3), (26, 22, 18), np.uint8)
            cv2.putText(bar, f"frame {i:4d}   {len(members)} parts tracked   "
                             f"dense model + 6-DoF pose",
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

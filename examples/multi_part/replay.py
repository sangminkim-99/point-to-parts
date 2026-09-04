"""Run the streaming part discovery over a recorded sequence.

Same code path the live camera demo uses, but fed from disk, so the state machine
and the achieved frame rate can be checked without hardware attached.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from examples.multi_part.streaming import Config, StreamingPartDiscovery
from point2pose.io.sources.dataset.sapien_reader import open_sequence
from point2pose.modules.tracker.tapir_tracker import TapirTracker
from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister)


def _naive_tail(s, cfg, args, r, a, gt):
    """The naive path shares the ground-truth report and nothing else."""
    if getattr(s, "unmodelled", None) is not None:
        print(f"[replay] unmodelled object surface: {100 * s.unmodelled:.0f}%")
    if gt is None or not s.parts:
        return
    from pathlib import Path as _P
    from examples.multi_part import evaluate as _ev
    res = _ev.report(r, a, gt["parts"], gt["lab"], s, gt["log"],
                     min_group=cfg.min_part_pts)
    try:
        res["joints"] = _ev.joint_metrics(r, gt["parts"], s, res["rows"],
                                          [f for f, _ in gt["log"]])
    except Exception as exc:
        print(f"[eval] joint metrics unavailable ({exc})")
    print("[eval] " + _ev.fmt(_P(args.seq_dir).name, res, len(s.parts)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--object-masks", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="debug/multi-parts/runs/stream/replay.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--hyp-every", type=int, default=None)
    ap.add_argument("--tapir-res", type=int, default=512)
    ap.add_argument("--co-sample", type=int, default=None)
    ap.add_argument("--min-inliers", type=int, default=None)
    ap.add_argument("--inlier-thres", type=float, default=None)
    ap.add_argument("--groupings", default=None,
                    help="subset of labels,coassoc,winsets,merge")
    ap.add_argument("--min-group", type=int, default=None)
    ap.add_argument("--min-group-frac", type=float, default=None)
    ap.add_argument("--mask-depth-jump", type=float, default=None)
    ap.add_argument("--mask-win", type=int, default=None)
    ap.add_argument("--outlier-r", type=float, default=None)
    ap.add_argument("--resplit", type=int, default=None)
    ap.add_argument("--resplit-wait", type=int, default=None)
    ap.add_argument("--resplit-min-ratio", type=float, default=None)
    ap.add_argument("--resplit-gain", type=float, default=None)
    ap.add_argument("--graph", type=int, default=None)
    ap.add_argument("--method", choices=["dense", "naive"], default="dense",
                    help="naive = sparse frontend (steps 1-3), layers optional")
    ap.add_argument("--dense", type=int, default=0,
                    help="naive only: step 4, per-part gaussian model")
    ap.add_argument("--refine", type=int, default=0,
                    help="naive only: step 5, rendering-based refinement")
    ap.add_argument("--split-out-frac", type=float, default=None)
    ap.add_argument("--ambiguous-band", type=float, default=None)
    ap.add_argument("--min-part-pts", type=int, default=None)
    ap.add_argument("--split-frames", type=int, default=None)
    ap.add_argument("--joint-track", type=int, default=None)
    ap.add_argument("--top-up", type=int, default=None)
    ap.add_argument("--mask-gate", type=int, default=None)
    ap.add_argument("--persist", type=int, default=None)
    ap.add_argument("--part-settle", type=int, default=None)
    ap.add_argument("--co-gap", type=float, default=None)
    ap.add_argument("--co-min-seen", type=float, default=None)
    ap.add_argument("--split-sep-sigma", type=float, default=None)
    ap.add_argument("--split-out-band", type=float, default=None)
    ap.add_argument("--split-bic", type=int, default=None)
    ap.add_argument("--cohort-veto", type=int, default=None)
    ap.add_argument("--merge-rigid", type=int, default=None)
    ap.add_argument("--merge-smooth", type=float, default=None)
    ap.add_argument("--axis-prior", type=float, default=None)
    ap.add_argument("--pending", type=int, default=None)
    ap.add_argument("--motion-gate", type=int, default=None)
    ap.add_argument("--motion-sigma", type=float, default=None)
    ap.add_argument("--motion-coherent", type=float, default=None)
    ap.add_argument("--motion-min-pts", type=int, default=None)
    ap.add_argument("--unc-gate", type=int, default=None)
    ap.add_argument("--unc-max", type=float, default=None)
    ap.add_argument("--rot-tol", type=float, default=None)
    ap.add_argument("--split-min-rot-deg", type=float, default=None)
    ap.add_argument("--split-min-trans-m", type=float, default=None)
    ap.add_argument("--config", default=None,
                    help="a file in configs/multi-part, or a path")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override any NaiveConfig field")
    ap.add_argument("--seed-prior", type=float, default=None)
    ap.add_argument("--max-parts", type=int, default=None)
    ap.add_argument("--key-view", type=int, default=None)
    ap.add_argument("--key-angle-deg", type=float, default=None)
    ap.add_argument("--min-live", type=int, default=None)
    ap.add_argument("--reseed-points", type=int, default=None)
    ap.add_argument("--reseed-gap", type=int, default=None)
    ap.add_argument("--frame-fallback", type=int, default=None)
    ap.add_argument("--frame-after", type=int, default=None)
    ap.add_argument("--sampler", default=None,
                    help="super_point_balanced | super_point_fps | super_point "
                         "| uniform_fps | orb | random")
    ap.add_argument("--occlude", default=None,
                    help="START:END:FRAC -- hide FRAC of the object's bounding "
                         "box (depth and mask) between those frames")
    ap.add_argument("--urdf", default=None,
                    help="write the discovered object out as a URDF")
    ap.add_argument("--resplit-cov-frac", type=float, default=None)
    ap.add_argument("--pips-iter", type=int, default=None)
    ap.add_argument("--grow-gate-rel", type=float, default=None)
    ap.add_argument("--outside-w", type=float, default=None)
    ap.add_argument("--view", type=int, default=1,
                    help="show a live cv2 window; 'q' quits, 'space' pauses")
    ap.add_argument("--hyp-panel", type=int, default=1,
                    help="strip of per-hypothesis render-vs-depth residuals")
    ap.add_argument("--models-dir", default=None,
                    help="RBO models/ directory (defaults beside sequences/)")
    ap.add_argument("--shorter-side", type=int, default=None)
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--regroup-every", type=int, default=None)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    args = ap.parse_args()

    # a RealSense recording (rgb/, depth/, cam_K.txt), an RBO sequence
    # (camera_rgb/, tf.csv) or a rendered SAPIEN sequence
    sd = Path(args.seq_dir).expanduser()
    if (sd / "cam_K.txt").exists():
        from examples.multi_part.recording import Recording
        r = Recording(args.seq_dir)
        rec_mode = True
    elif (sd / "camera_rgb").is_dir():
        from point2pose.io.sources.dataset.rbo_reader import RBOReader
        r = RBOReader(str(sd), models_dir=args.models_dir,
                      shorter_side=args.shorter_side)
        rec_mode = False
    else:
        r = open_sequence(args.seq_dir)
        rec_mode = False
    ext = None
    if args.object_masks:
        d = np.load(args.object_masks, allow_pickle=True)
        ext = {int(f): (d["masks"][k].max(axis=0) > 0).astype(np.uint8)
               for k, f in enumerate(d["frames"]) if k < len(d["masks"])}

    def object_mask(i):
        if ext is not None:
            return ext.get(i)
        if rec_mode:
            return r.get_mask(i)
        m = np.zeros((r.H, r.W), np.uint8)
        for x in r.get_masks(i):
            m |= x
        return m

    frames = list(range(0, len(r), args.stride))
    if ext is not None:
        frames = [i for i in frames if ext.get(i) is not None]
    if rec_mode:
        frames = [i for i in frames if object_mask(i) is not None]
        if not frames:
            raise SystemExit(
                "no masks: run  python -m examples.multi_part.annotate "
                f"--seq-dir {args.seq_dir}")
    if args.max_frames:
        frames = frames[:args.max_frames]

    cfg = Config()
    if args.n_points:
        cfg.n_points = args.n_points
    if args.sampler:
        cfg.sampler = args.sampler
    if args.hyp_every:
        cfg.hyp_every = args.hyp_every
    if args.co_sample:
        cfg.co_sample = args.co_sample
    if args.min_inliers:
        cfg.min_inliers = args.min_inliers
    if args.inlier_thres:
        cfg.inlier_thres = args.inlier_thres
    if args.groupings:
        cfg.groupings = args.groupings
    if args.min_group:
        cfg.min_group = args.min_group
    if args.outlier_r is not None:
        cfg.outlier_r = args.outlier_r
    if args.mask_win is not None:
        cfg.mask_win = args.mask_win
    if args.mask_depth_jump is not None:
        cfg.mask_depth_jump = args.mask_depth_jump
    if args.min_group_frac is not None:
        cfg.min_group_frac = args.min_group_frac
    if args.resplit is not None:
        cfg.resplit = bool(args.resplit)
    if args.resplit_wait:
        cfg.resplit_wait = args.resplit_wait
    if args.graph is not None:
        cfg.graph = bool(args.graph)
    if args.resplit_gain is not None:
        cfg.resplit_gain = args.resplit_gain
    if args.resplit_min_ratio is not None:
        cfg.resplit_min_ratio = args.resplit_min_ratio
    if args.resplit_cov_frac is not None:
        cfg.resplit_cov_frac = args.resplit_cov_frac
    if args.pips_iter:
        cfg.num_pips_iter = args.pips_iter
    if args.grow_gate_rel is not None:
        cfg.grow_gate_rel = args.grow_gate_rel
    if args.outside_w is not None:
        cfg.outside_w = args.outside_w
    if args.min_frames:
        cfg.min_frames_before_split = args.min_frames
    if args.regroup_every:
        cfg.regroup_every = args.regroup_every
    tracker = TapirTracker({"checkpoint_path": args.checkpoint,
                            "resize_height": args.tapir_res, "resize_width": args.tapir_res,
                            "num_pips_iter": cfg.num_pips_iter,
                            "visible_threshold": 0.5, "device": "cuda"})
    reg = SVDClusterRANSACRegister({
        "ransac_iters": cfg.ransac_iters, "sample_size": 4,
        "inlier_thres": cfg.inlier_thres, "min_inliers": cfg.min_inliers,
        "max_clusters": cfg.max_hyp, "use_uncertainty": False})

    if args.method == "naive":
        from examples.multi_part.naive import NaiveConfig, NaivePartTracker
        from examples.multi_part.acquire import apply_config
        # NaiveConfig owns its own defaults; injecting the dense Config's here
        # meant the file said one thing and every run did another
        ncfg = NaiveConfig(dense=bool(args.dense),
                           refine=bool(args.dense and args.refine))
        apply_config(ncfg, args.config)
        for k, v in (("split_out_frac", args.split_out_frac),
                     ("ambiguous_band", args.ambiguous_band),
                     ("min_part_pts", args.min_part_pts),
                     ("split_frames", args.split_frames),
                     ("joint_track", None if args.joint_track is None
                      else bool(args.joint_track)),
                     ("top_up", None if args.top_up is None else bool(args.top_up)),
                     ("mask_gate", None if args.mask_gate is None
                      else bool(args.mask_gate)),
                     ("persist", None if args.persist is None
                      else bool(args.persist)),
                     ("part_settle", args.part_settle),
                     ("co_gap", args.co_gap),
                     ("co_min_seen", args.co_min_seen),
                     ("split_sep_sigma", args.split_sep_sigma),
                     ("split_out_band", args.split_out_band),
                     ("split_bic", None if args.split_bic is None
                      else bool(args.split_bic)),
                     ("merge_rigid", None if args.merge_rigid is None
                      else bool(args.merge_rigid)),
                     ("merge_smooth", args.merge_smooth),
                     ("axis_prior", args.axis_prior),
                     ("pending", None if args.pending is None
                      else bool(args.pending)),
                     ("motion_gate", None if args.motion_gate is None
                      else bool(args.motion_gate)),
                     ("motion_sigma", args.motion_sigma),
                     ("motion_coherent", args.motion_coherent),
                     ("motion_min_pts", args.motion_min_pts),
                     ("unc_gate", None if args.unc_gate is None
                      else bool(args.unc_gate)),
                     ("unc_max", args.unc_max),
                     ("rot_tol", args.rot_tol),
                     ("split_min_rot_deg", args.split_min_rot_deg),
                     ("split_min_trans_m", args.split_min_trans_m),
                     ("seed_prior", args.seed_prior),
                     ("max_parts", args.max_parts),
                     ("cohort_veto", None if args.cohort_veto is None
                      else bool(args.cohort_veto)),
                     ("key_view", None if args.key_view is None
                      else bool(args.key_view)),
                     ("key_angle_deg", args.key_angle_deg),
                     ("min_live", args.min_live),
                     ("reseed_points", args.reseed_points),
                     ("reseed_gap", args.reseed_gap),
                     ("frame_fallback", None if args.frame_fallback is None
                      else bool(args.frame_fallback)),
                     ("frame_after", args.frame_after),
                     ("sampler", args.sampler)):
            if v is not None:
                setattr(ncfg, k, v)
        if args.n_points:
            ncfg.n_points = args.n_points
        if args.sampler:
            ncfg.sampler = args.sampler
        from examples.multi_part.acquire import apply_overrides
        apply_overrides(ncfg, args.set)
        s = NaivePartTracker(r.K, ncfg, tracker, reg)
        cfg = ncfg
    else:
        s = StreamingPartDiscovery(r.K, cfg, tracker, reg)
    a = frames[0]
    s.start(r.get_color(a), r.get_depth(a), object_mask(a))
    if args.method == "naive":
        print(f"[replay] naive: {len(s.anchor_xyz)} points"
              + (f", {len(s.model.cloud)} gaussians" if s.model else "")
              + f", {len(frames)} frames")
    else:
        print(f"[replay] {len(s.cloud)} gaussians, stride {s.gauss_stride}, "
              f"{len(frames)} frames")

    # ground truth, when the reader has it
    gt = None
    if not rec_mode and hasattr(r, "render_part_index_map"):
        try:
            from examples.multi_part import evaluate as _ev
            lab = (_ev.gt_labels_at(r, a, s.pts0) if args.method == "naive"
                   else _ev.gt_labels(r, a, r.get_depth(a), object_mask(a),
                                      s.gauss_stride))
            gt = {"parts": list(r.get_object_names()), "lab": lab, "log": []}
        except Exception as exc:
            print(f"[replay] no GT evaluation ({exc})")
            gt = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer, times, breakdown, split_costs = None, [], {}, []
    occ = None
    if args.occlude:
        a0, a1, fr = args.occlude.split(":")
        occ = (int(a0), int(a1), float(fr))
        print(f"[replay] occluding {100 * occ[2]:.0f}% of the object "
              f"from frame {occ[0]} to {occ[1]}")

    for i in frames[1:]:
        rgb, dep, m = r.get_color(i), r.get_depth(i), object_mask(i)
        if m is None:
            continue
        if occ is not None and occ[0] <= i < occ[1]:
            ys, xs = np.where(m > 0)
            if len(xs):
                # hide the right-hand slice of the object, sensor and all
                cut = int(xs.min() + (1.0 - occ[2]) * (xs.max() - xs.min()))
                m = m.copy(); m[:, cut:] = 0
                dep = dep.copy(); dep[:, cut:] = 0
                rgb = rgb.copy(); rgb[:, cut:] = 0
        s.step(rgb, dep, m)
        times.append(s.last_timings["total_ms"])
        if getattr(s, "last_split_ms", None):
            split_costs.append(s.last_split_ms); s.last_split_ms = None
        for k, v in s.last_timings.items():
            breakdown.setdefault(k, []).append(v)
        if gt is not None and s.parts:
            gt["log"].append((int(i), [None if p.pose is None else p.pose.copy()
                                       for p in s.parts]))

        vis = s.render(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy())
        if args.hyp_panel and hasattr(s, "hypothesis_panel"):
            panel = s.hypothesis_panel(dep, m, width=vis.shape[1] // 4)
            if panel is not None:
                vis = np.vstack([vis, s.fit_panel(panel, vis.shape[1])])
        bar = np.full((30, vis.shape[1], 3), (28, 24, 20), np.uint8)
        fps = 1000.0 / max(np.median(times[-30:]), 1e-6)
        d = getattr(s, "diag", {})
        if args.method == "naive":
            label = (f"{s.state} parts {len(s.parts)} | resid "
                     + " ".join(f"{p.resid*1000:.0f}" for p in s.parts)
                     + f" mm | g {len(s.model.cloud) if s.model else 0} "
                     f"| {fps:.1f} fps")
        else:
            label = (f"{s.state} parts {len(s.parts)} | hyp {d.get('hyp', 0)} "
                     f"tracks {d.get('tracks_live', 0)}/{d.get('tracks', 0)} "
                     f"decisive {100*d.get('decisive', 0):.0f}% "
                     f"tries {d.get('tries', 0)} | g {d.get('gauss', 0)//1000}k "
                     f"ws {d.get('winsets', 0)} | {fps:.1f} fps")
        cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([vis, bar])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     15, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

        if args.view:
            cv2.imshow("multi-part replay", canvas)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                print("[replay] stopped by user")
                break
            if k == ord(" "):                       # pause until space again
                while (cv2.waitKey(50) & 0xFF) != ord(" "):
                    pass
    if writer:
        writer.release()
    if args.view:
        cv2.destroyAllWindows()

    t = np.array(times)
    print(f"[replay] wrote {args.out}")
    print(f"[replay] per-frame {np.median(t):.1f} ms median, "
          f"{np.percentile(t, 90):.1f} ms p90  ->  {1000/np.median(t):.1f} fps")
    for k, v in breakdown.items():
        print(f"[replay]   {k:12s} median {np.median(v):7.1f} ms")
    if split_costs:
        print(f"[replay]   split attempt  median {np.median(split_costs):7.1f} ms "
              f"({len(split_costs)} attempts)")
    try:
        import resource
        print(f"[replay] peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f} GB, "
              f"cloud {len(s.cloud)} gaussians, "
              f"winsets {len(getattr(s.assign, 'winsets', []))}")
    except Exception:
        pass
    if args.method == "naive":
        c = getattr(s, "controllable", None)
        fps_med = 1000.0 / max(np.median(times), 1e-6)
        if c:
            print(f"[replay] time to controllable: frame {c['frame']} "
                  f"({c['frame'] / max(fps_med, 1e-6):.1f} s at {fps_med:.0f} fps"
                  + (f", split at {c['split_frame']}" if c['split_frame'] else "")
                  + f") -- {c['kind']}, conf {c['conf']:.2f}, "
                  f"axis +-{c['axis_std_deg']:.1f} deg")
        else:
            print("[replay] never became controllable")
        for j, p in enumerate(s.parts):
            if p.joint is not None and p.joint.kind:
                cc = p.joint.confidence()
                print(f"[replay]   part {j} joint {p.joint.kind}: "
                      f"conf {cc['conf']:.2f} (type {cc['type_p']:.2f}, "
                      f"axis +-{cc['axis_std_deg']:.1f} deg, "
                      f"watched {(np.degrees(cc['span']) if p.joint.kind == 'revolute' else 1000 * cc['span']):.0f}"
                      f"{' deg' if p.joint.kind == 'revolute' else ' mm'}"
                      f"{'' if cc['excitation'] >= 1.0 else ' (short)'}, "
                      f"joint? {'yes' if cc.get('valid') else 'NO'} "
                      f"(smooth {cc.get('smooth', float('nan')):.2f}), "
                      f"axis {100 * getattr(p.joint, 'axis_far', 0.0):.0f} cm "
                      f"from the object)")
        if getattr(s, "small_blocked", 0):
            print(f"[replay] {s.small_blocked} splits refused: the relative "
                  f"motion was too small to be a joint")
        if getattr(s, "unc_dropped", 0):
            print(f"[replay] {s.unc_dropped} track-frames ignored: the tracker "
                  f"was not sure of them")
        if getattr(s, "still_frames", 0):
            print(f"[replay] {s.still_frames} part-frames carried no evidence: "
                  f"nothing moved that one rigid body could not explain")
        if getattr(s, "placed_total", 0) or getattr(s, "dropped_total", 0):
            print(f"[replay] {getattr(s, 'placed_total', 0)} new tracks placed "
                  f"by the motion they follow, "
                  f"{getattr(s, 'dropped_total', 0)} never decided")
        if getattr(s, "merged_total", 0):
            print(f"[replay] {s.merged_total} parts merged back: their joint "
                  f"never opened")
        if getattr(s, "cohort_blocked", 0):
            print(f"[replay] {s.cohort_blocked} splits refused: the groups were "
                  f"two seeding batches, not two parts")
        if getattr(s, "bic_blocked", 0):
            print(f"[replay] {s.bic_blocked} splits refused: not worth six more "
                  f"parameters")
        print(f"[replay] naive: {len(s.parts)} parts, splits {s.split_log}"
              + f", joint-tracked {getattr(s, 'joint_frames', 0)} part-frames"
              + (f", grown {getattr(s, 'grown', 0)}, carved "
                 f"{getattr(s, 'carved', 0)}, relabelled {getattr(s, 'moved', 0)}"
                 if s.model else ""))
        if args.urdf:
            from examples.multi_part import urdf_export as _ux
            gm = (s.model.cloud.means.detach().cpu().numpy()
                  if s.model else None)
            if gm is None:
                pts_of = lambda j: s.anchor_xyz[s.parts[j].idx]
            else:
                pts_of = lambda j: gm[s.model.labels[:len(gm)] == j]
            out, w = _ux.export(args.urdf, s.parts, pts_of)
            print(f"[replay] wrote {out} with {len(w)} collision meshes")
        _naive_tail(s, cfg, args, r, a, gt)
        return
    print(f"[replay] growth: {getattr(s, 'grown_total', 0)} gaussians added over "
          f"{getattr(s, 'grow_calls', 0)} attempts, "
          f"{getattr(s, 'grow_blocked', 0)} blocked by the energy gate "
          f"(rel gate {cfg.grow_gate_rel}); part energies "
          f"{[round(p.energy, 4) for p in s.parts]}")
    n0 = getattr(s, "n_initial", None)
    if n0:
        import numpy as _np
        gm = s.cloud.means.detach().cpu().numpy()
        for j, pt in enumerate(s.parts):
            w = pt.weights[:len(gm)] > 0.5
            orig = int(w[:n0].sum()); grown = int(w[n0:].sum())
            ext = _np.round(gm[w].max(0) - gm[w].min(0), 3) if w.any() else None
            print(f"[replay]   part {j}: {orig} original + {grown} grown, extent {ext}")
    if getattr(s, "unmodelled", None) is not None:
        print(f"[replay] unmodelled object surface: {100 * s.unmodelled:.0f}%")
    if getattr(s, "graph_used", 0):
        print(f"[replay] pose graph accepted {s.graph_used} part-frames")
    if getattr(s, "pruned_total", 0):
        print(f"[replay] pruned {s.pruned_total} gaussians no part explained")
    if gt is not None and s.parts:
        from examples.multi_part import evaluate as _ev
        res = _ev.report(r, a, gt["parts"], gt["lab"], s, gt["log"],
                         min_group=(cfg.min_part_pts if args.method == "naive"
                                    else cfg.min_group))
        try:
            res["joints"] = _ev.joint_metrics(r, gt["parts"], s, res["rows"],
                                              [f for f, _ in gt["log"]])
        except Exception as exc:
            print(f"[eval] joint metrics unavailable ({exc})")
        print("[eval] " + _ev.fmt(Path(args.seq_dir).name, res, len(s.parts)))
    print(f"[replay] final state: {s.state}, {len(s.parts)} parts")
    for j, p in enumerate(s.parts):
        jm = p.joint
        kind = jm.kind if jm is not None else None
        print(f"           part {j}: {int((p.weights > 0.5).sum())} gaussians, "
              f"energy {p.energy:.4f}, joint {kind}")


if __name__ == "__main__":
    main()

"""Run the real Point2Pose pipeline on an RBO sequence, with part discovery.

Unlike experiments/articulated/demo_part_discovery.py -- which drives TAPIR and
the register directly and reimplements the front end -- this runs the actual
ModularPipeline, so a discovered part inherits everything the pipeline provides:
its own keyframes and TSDF, the pose jump guard, cluster selection against the
dense map, uncertainty-weighted registration and the local graph optimizer.

The object starts as ONE rigid body (the union of the GT part masks stands in
for a user's click). Parts appear only when the object articulates.
"""

import argparse
import json
import os

# TAPIR keeps per-point state, and every spawned part makes the sampler add a
# fresh batch of keypoints, so total tracks grow with the part count. On a 12 GB
# card that reached OOM mid-sequence; expandable segments avoids the
# fragmentation that made it fail earlier than the true limit.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
import time

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.data_types.frame import Frame
from point2pose.io.sources.dataset.sapien_reader import open_sequence
from point2pose.pipeline.modular_pipeline import ModularPipeline
from point2pose.utils.visualization import draw_oriented_3d_box
from experiments.articulated.demo_part_discovery import fit_oriented_box

PART_COLORS = [(66, 135, 245), (214, 130, 60), (90, 190, 100),
               (170, 100, 220), (80, 200, 230), (120, 120, 250)]


def union_mask(reader, i, depth, mad_k=3.0, min_tol=0.05, border=4):
    """Whole-object mask, gated to the dominant depth mode.

    The GT part masks are raycast from meshes and so are amodal: they cover
    pixels where a hand or the scene's pole is in front of the object, whose
    depth then belongs to the occluder. Those pixels lift to badly wrong 3D
    points, so they are removed here rather than poisoning initialization.
    """
    m = np.zeros((reader.H, reader.W), np.uint8)
    for mm in reader.get_masks(i):
        m |= mm
    m = cv2.erode(m, np.ones((border, border), np.uint8))
    z = depth[m > 0]
    z = z[z > 0.1]
    if z.size:
        med = float(np.median(z))
        tol = max(mad_k * 1.4826 * float(np.median(np.abs(z - med))), min_tol)
        m = (m > 0) & (np.abs(depth - med) <= tol) & (depth > 0.1)
        m = m.astype(np.uint8)
    return m


def draw_overlay(rgb, pipeline, reader, frame_id, gt_label, box_cache):
    """Draw each object's 3D box and its tracked points.

    Filled dot = the tracker reports the point visible; hollow ring = occluded.
    Occluded points are still drawn so the occlusion is legible, but they take
    no part in registration.
    """
    vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    K = reader.K
    tt = pipeline.track_table
    n = len(pipeline.objects)
    H, W = vis.shape[:2]

    for oid, obj in enumerate(pipeline.objects):
        color = PART_COLORS[oid % len(PART_COLORS)]

        # box: fitted once per object in its own frame, then carried by its pose
        if oid not in box_cache and obj.key_points is not None \
                and obj.key_points.shape[0] >= 8:
            try:
                box_cache[oid] = fit_oriented_box(obj.key_points)
            except Exception:
                box_cache[oid] = None
        box = box_cache.get(oid, None)
        if box is not None and obj.pose is not None:
            pts_cam = (obj.pose @ np.hstack([box, np.ones((8, 1))]).T).T[:, :3]
            if np.all(pts_cam[:, 2] > 1e-6):
                vis = draw_oriented_3d_box(K, vis, obj.pose, box,
                                           line_color=color, linewidth=2)
                uv = (K @ pts_cam.T).T
                uv = uv[:, :2] / uv[:, 2:3]
                tag = "object (rigid)" if (oid == 0 and n == 1) else (
                    "base" if oid == 0 else f"part {oid}")
                x0 = int(np.clip(uv[:, 0].min(), 0, W - 110))
                y0 = int(np.clip(uv[:, 1].min(), 18, H - 4))
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x0, y0 - th - 8), (x0 + tw + 8, y0), color, -1)
                cv2.putText(vis, tag, (x0 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (20, 20, 20), 1, cv2.LINE_AA)

        idxs = tt.obj2track_map.get(oid, None)
        if idxs is None or len(idxs) == 0:
            continue
        idxs = np.asarray(idxs).reshape(-1)
        idxs = idxs[idxs < len(tt.track_2d)]
        for t in idxs:
            p = tt.track_2d[t]
            if not np.all(np.isfinite(p)):
                continue
            pt = (int(round(p[0])), int(round(p[1])))
            if not (0 <= pt[0] < W and 0 <= pt[1] < H):
                continue
            if bool(tt.visible[t]):
                cv2.circle(vis, pt, 3, color, -1, cv2.LINE_AA)
            else:
                cv2.circle(vis, pt, 3, color, 1, cv2.LINE_AA)

    n_vis = int(np.sum(tt.visible)) if len(tt.visible) else 0
    n_occ = int(len(tt.visible) - n_vis)
    bar = np.full((34, W, 3), (28, 24, 20), np.uint8)
    status = f"{n} parts tracked" if n > 1 else "single rigid body"
    cv2.putText(bar, f"frame {frame_id:04d}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, status, (140, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (120, 220, 140) if n > 1 else (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(bar, f"{n_occ} occluded", (W - 400, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.circle(bar, (W - 250, 17), 3, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.putText(bar, "visible", (W - 240, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(bar, (W - 160, 17), 3, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, "occluded", (W - 150, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    return np.vstack([vis, bar])


def find_init_frame(reader, frames, min_intensity=30.0, min_lapvar=150.0,
                    mask_erode=4, max_scan=60):
    """First frame usable for initialization.

    RBO sequences can start before the camera is exposing -- cabinet04 and
    cabinet05 open on frames that are black inside the object mask -- and the
    auto white balance takes a while to settle. Initializing there yields almost
    no keypoints, and the object is then unregisterable for the whole run, which
    looks exactly like a discovery failure but is not one.
    """
    for k, i in enumerate(frames[:max_scan]):
        depth = reader.get_depth(i)
        m = union_mask(reader, i, depth, border=mask_erode)
        if m.sum() < 500:
            continue
        gray = cv2.cvtColor(reader.get_color(i), cv2.COLOR_RGB2GRAY)
        sel = m > 0
        if gray[sel].mean() < min_intensity:
            continue
        if cv2.Laplacian(gray, cv2.CV_64F)[sel].var() < min_lapvar:
            continue
        return k
    return 0


def part_region_mask(pipeline, oid, object_mask, dilate_px=25):
    """Region a part is allowed to claim new keypoints from.

    Model-free: the convex hull of the part's own visible tracks, dilated a
    little and clipped to the object. Without this a part keeps absorbing points
    from the rest of the object and its purity collapses.
    """
    tt = pipeline.track_table
    idxs = tt.obj2track_map.get(oid, None)
    if idxs is None or len(idxs) == 0:
        return np.zeros_like(object_mask)
    idxs = np.asarray(idxs).reshape(-1)
    idxs = idxs[idxs < len(tt.track_2d)]
    pts = tt.track_2d[idxs]
    ok = np.all(np.isfinite(pts), axis=1) & tt.visible[idxs]
    pts = pts[ok]
    if pts.shape[0] < 3:
        return np.zeros_like(object_mask)

    region = np.zeros_like(object_mask)
    hull = cv2.convexHull(pts.astype(np.float32).reshape(-1, 1, 2))
    cv2.fillConvexPoly(region, hull.astype(np.int32), 1)
    if dilate_px > 0:
        k = np.ones((dilate_px, dilate_px), np.uint8)
        region = cv2.dilate(region, k)
    return (region & (object_mask > 0)).astype(np.uint8)


def label_new_tracks(reader, frame_idx, pipeline, gt_label):
    """Assign each newly created track the GT part it was born on.

    Labels are taken at the frame the track first appears, since the pipeline
    keeps adding keypoints as the object is explored. Used only for scoring --
    never to drive discovery.
    """
    tt = pipeline.track_table
    n = len(tt.track_2d)
    if n <= gt_label.size:
        return gt_label
    grown = np.full(n, -2, dtype=np.int16)     # -2 = not yet labelled
    grown[: gt_label.size] = gt_label
    new = np.where(grown == -2)[0]
    if new.size:
        pim = reader.render_part_index_map(frame_idx)
        H, W = pim.shape
        for t in new:
            p = tt.track_2d[t]
            if not np.all(np.isfinite(p)):
                grown[t] = -1
                continue
            x = int(np.clip(round(p[0]), 0, W - 1))
            y = int(np.clip(round(p[1]), 0, H - 1))
            grown[t] = int(pim[y, x])
    return grown


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _relocate_checkpoints(node, out_root=None):
    """Re-anchor the config's absolute paths onto this checkout.

    The shipped configs carry the original author's home directory for every
    checkpoint and debug directory, which is unwritable here.
    """
    from omegaconf import DictConfig, ListConfig

    if isinstance(node, DictConfig):
        for k in list(node.keys()):
            v = node[k]
            if isinstance(v, str) and "/checkpoints/" in v:
                node[k] = os.path.join(
                    REPO_ROOT, "checkpoints", v.split("/checkpoints/", 1)[1]
                )
            elif isinstance(v, str) and "/point-to-pose/" in v and v.startswith("/"):
                tail = v.split("/point-to-pose/", 1)[1]
                node[k] = os.path.join(out_root or REPO_ROOT, tail)
            else:
                _relocate_checkpoints(v, out_root)
    elif isinstance(node, ListConfig):
        for v in node:
            _relocate_checkpoints(v, out_root)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--config", default="configs/ycbinisaac/ycbinisaac_single.yaml")
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--object-masks-dir", default=None,
                    help="directory holding <sequence>_masks.npz; lets a batch run "
                         "pick up the right mask file per sequence")
    ap.add_argument("--object-masks", default=None,
                    help="npz from run_sam2_propagate.py --save-masks. Use the "
                         "SAM2-propagated whole-object mask instead of one built "
                         "from GT parts, so no ground truth enters the input path.")
    ap.add_argument("--out", default="debug/multi-parts/pipeline_part_discovery.mp4")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--inlier-thres", type=float, default=None)
    ap.add_argument("--detector",
                    choices=["consensus", "trajectory", "rigidity", "window_ransac"],
                    default="consensus",
                    help="consensus = per-frame RANSAC set matching; "
                         "trajectory = cluster per-track motion fits (VideoArtGS-style); "
                         "rigidity = pairwise distance-constancy affinity + spectral "
                         "clustering (Brox & Malik ECCV 2010), invariant to common motion; "
                         "window_ransac = sequential RANSAC SE(3) over a time window")
    ap.add_argument("--persist", type=int, default=6)
    ap.add_argument("--overlap-thres", type=float, default=0.4,
                    help="membership Jaccard needed to match a candidate across frames")
    ap.add_argument("--max-clusters", type=int, default=None,
                    help="sequential RANSAC cluster cap; the config default of 10 "
                         "fragments into many small, unstable sets")
    ap.add_argument("--min-rel-trans", type=float, default=0.015)
    ap.add_argument("--max-parts", type=int, default=4,
                    help="max parts spawned from any single object")
    ap.add_argument("--max-parts-total", type=int, default=5,
                    help="max parts spawned overall, since parts can spawn parts")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--max-depth", type=float, default=None,
                    help="pipeline depth cut-off; the shipped config uses 2 m")
    ap.add_argument("--num-points", type=int, default=None,
                    help="keypoints sampled per object; the shipped config uses 30, "
                         "which is sparse for part discovery")
    ap.add_argument("--min-separation-px", type=float, default=None,
                    help="lower this for small/thin objects (config default 5)")
    ap.add_argument("--edge-margin-px", type=int, default=None,
                    help="lower this for small/thin objects (config default 2)")
    ap.add_argument("--part-mask-dilate", type=int, default=25,
                    help="dilation of a part's own region, in pixels")
    ap.add_argument("--whole-object-masks", action="store_true",
                    help="give every part the full object mask (the old, leaky behaviour)")
    ap.add_argument("--auto-init-frame", action="store_true", default=True,
                    help="skip leading frames that are dark or untextured")
    ap.add_argument("--no-auto-init-frame", dest="auto_init_frame",
                    action="store_false")
    ap.add_argument("--mask-erode", type=int, default=4,
                    help="erosion applied to the initial mask")
    ap.add_argument("--traj-window", type=int, default=60)
    ap.add_argument("--traj-min-history", type=int, default=12)
    ap.add_argument("--traj-static-thres", type=float, default=0.02)
    ap.add_argument("--traj-fit-thres", type=float, default=0.01)
    ap.add_argument("--traj-max-k", type=int, default=4,
                    help="max K-means clusters per motion type")
    ap.add_argument("--traj-min-silhouette", type=float, default=0.45,
                    help="silhouette needed to accept a split into >1 part")
    ap.add_argument("--traj-voxel-rel", type=float, default=0.02,
                    help="adaptive voxel size as a fraction of trajectory range")
    ap.add_argument("--traj-update-every", type=int, default=2)
    ap.add_argument("--traj-min-points", type=int, default=12)
    ap.add_argument("--rig-stable", type=int, default=3,
                    help="matched detections needed before a group becomes a part")
    ap.add_argument("--rig-match-overlap", type=float, default=0.6,
                    help="membership Jaccard needed to call it the same candidate")
    ap.add_argument("--rig-min-ratio", type=float, default=4.0,
                    help="between-group std must exceed within-group std by this factor")
    ap.add_argument("--rig-sigma", type=float, default=0.010,
                    help="affinity bandwidth on pairwise distance std, metres")
    ap.add_argument("--rig-tol", type=float, default=0.012,
                    help="pair counts as rigid below this distance std, metres")
    ap.add_argument("--rig-min-cross", type=float, default=0.020,
                    help="minimum between-group distance std to accept a split")
    ap.add_argument("--work-dir", default=None,
                    help="where the pipeline writes its debug output")
    args = ap.parse_args()

    reader = open_sequence(args.seq_dir, models_dir=args.models_dir)
    gt_parts = reader.get_object_names()

    ext_masks = None
    if args.object_masks is None and args.object_masks_dir:
        cand = os.path.join(args.object_masks_dir,
                            f"{reader.get_video_name()}_masks.npz")
        if os.path.exists(cand):
            args.object_masks = cand
        else:
            print(f"[run] no mask file at {cand}; falling back to GT-derived masks")
    if args.object_masks:
        d = np.load(args.object_masks, allow_pickle=True)
        # collapse whatever objects SAM2 tracked into one whole-object mask
        ext_masks = {int(f): (d["masks"][k].max(axis=0) > 0).astype(np.uint8)
                     for k, f in enumerate(d["frames"]) if k < len(d["masks"])}
        print(f"[run] using SAM2 object masks for {len(ext_masks)} frames "
              f"from {args.object_masks} (no GT in the input path)")
    print(f"[run] {reader.get_video_name()}  {len(reader)} frames  GT parts {gt_parts}")

    cfg = OmegaConf.load(args.config)
    work_dir = args.work_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.out)) or ".", "pipeline_work"
    )
    os.makedirs(work_dir, exist_ok=True)
    _relocate_checkpoints(cfg, out_root=work_dir)
    cfg.pipeline.params.max_num_obj = 1          # start as ONE rigid body
    cfg.pipeline.params.use_segmenter = False
    cfg.pipeline.params.save_meta_data = False
    cfg.pipeline.params.save_pose = False
    if args.inlier_thres is not None:
        cfg.register.params.inlier_thres = args.inlier_thres
    if args.min_separation_px is not None:
        cfg.sampler.params.min_separation_px = args.min_separation_px
    if args.edge_margin_px is not None:
        cfg.sampler.params.edge_margin_px = args.edge_margin_px
    if args.max_depth is not None:
        cfg.pipeline.params.max_depth = args.max_depth
    if args.max_clusters is not None:
        cfg.register.params.max_clusters = args.max_clusters
    if args.num_points is not None:
        cfg.sampler.params.num_points = args.num_points
        cfg.sampler.params.max_points = max(
            args.num_points, int(cfg.sampler.params.get("max_points", 50))
        )
    cfg.part_discovery = OmegaConf.create({
        "type": args.detector,
        "params": {
            "enabled": True,
            "persist_frames": args.persist,
            "overlap_thres": args.overlap_thres,
            "min_rel_trans": args.min_rel_trans,
            "max_parts_per_object": args.max_parts,
            "max_parts_total": args.max_parts_total,
            "verbose": True,
            "window": args.traj_window,
            "min_history": args.traj_min_history,
            "static_thres": args.traj_static_thres,
            "fit_thres": args.traj_fit_thres,
            "max_k": args.traj_max_k,
            "min_silhouette": args.traj_min_silhouette,
            "voxel_rel": args.traj_voxel_rel,
            "update_every": args.traj_update_every,
            "min_points": args.traj_min_points,
            "sigma": args.rig_sigma,
            "rigid_tol": args.rig_tol,
            "min_cross_std": args.rig_min_cross,
            "stable_updates": args.rig_stable,
            "match_overlap": args.rig_match_overlap,
            "min_ratio": args.rig_min_ratio,
        }
    })

    pipeline = ModularPipeline(cfg)

    frames = list(range(0, len(reader), args.stride))
    if args.auto_init_frame and ext_masks is None:
        start = find_init_frame(reader, frames, mask_erode=args.mask_erode)
    elif ext_masks is not None:
        # the SAM2 run already skipped unusable leading frames
        start = next((k for k, i in enumerate(frames) if i in ext_masks), 0)
    else:
        start = 0
    if start:
        print(f"[run] skipping {start} unusable leading frames "
              f"(dark / untextured); initializing on frame {frames[start]}")
    frames = frames[start:]
    if args.max_frames:
        frames = frames[: args.max_frames]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer, history = None, []
    gt_label = np.full(0, -2, dtype=np.int16)
    box_cache = {}

    for n, i in enumerate(frames):
        rgb, depth = reader.get_color(i), reader.get_depth(i)
        if ext_masks is not None:
            m = ext_masks.get(i)
            if m is None:
                continue                      # no SAM2 mask for this frame
            if args.mask_erode > 0:
                m = cv2.erode(m, np.ones((args.mask_erode,) * 2, np.uint8))
        else:
            m = union_mask(reader, i, depth, border=args.mask_erode)
        if m.sum() < 200:
            continue

        # Per-part masks. Giving every part a copy of the whole-object mask lets
        # the sampler add points from anywhere on the object to any part, which
        # measurably destroys purity: parts that grew that way fell to ~50%
        # while parts that did not stayed at 100%. So each part is confined to
        # the region its own tracks occupy, and only the base keeps the full
        # mask (it owns everything not yet claimed).
        n_obj = len(pipeline.objects) + args.max_parts + 1
        stack = np.repeat(m[None], n_obj, axis=0)
        if not args.whole_object_masks:
            for oid in range(1, len(pipeline.objects)):
                stack[oid] = part_region_mask(pipeline, oid, m, args.part_mask_dilate)
        mask = torch.from_numpy(stack).unsqueeze(1).to(device)

        frame = Frame(id=i, rgb=rgb, depth=depth, mask=mask,
                      intrinsics=reader.K, depth_factor=1.0, timestamp=time.time())

        if n == 0:
            pipeline.initialize_first_frame(frame)
        else:
            pipeline.step(frame)

        gt_label = label_new_tracks(reader, i, pipeline, gt_label)
        canvas = draw_overlay(rgb, pipeline, reader, i, gt_label, box_cache)
        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(canvas)

        for oid, obj in enumerate(pipeline.objects):
            k = obj.key_points.shape[0] if obj.key_points is not None else 0
            if oid not in box_cache or abs(k - box_cache.get(("n", oid), k)) > 20:
                box_cache.pop(oid, None)
                box_cache[("n", oid)] = k

        history.append({"frame": int(i), "n_objects": len(pipeline.objects),
                        "joint_states": reader.get_joint_states(i)})
        if n % 25 == 0:
            print(f"  frame {i:4d}  objects={len(pipeline.objects)}  "
                  f"kp={[o.key_points.shape[0] for o in pipeline.objects]}")

    if writer:
        writer.release()
    print(f"\n[run] wrote {args.out}")
    print(f"[run] finished with {len(pipeline.objects)} objects (GT parts {len(gt_parts)})")
    for s in pipeline.part_spawn_log:
        js = reader.get_joint_states(s["frame"]) if s["frame"] is not None else {}
        jtype = {j["name"]: j["type"] for j in reader.joints}
        parts_txt = ", ".join(
            f"{k}={np.degrees(v):.0f}deg" if jtype.get(k) == "revolute"
            else f"{k}={v*1000:.0f}mm"
            for k, v in js.items()
        )
        print(f"  spawn frame {s['frame']}: part {s['child']} from {s['parent']}, "
              f"{s['n_points']} pts | joint states " + parts_txt)

    # ---- score the discovered decomposition against ground truth ----
    print(f"\n[run] per-object composition (GT parts {gt_parts}):")
    scoring = []
    for oid, obj in enumerate(pipeline.objects):
        idxs = pipeline.track_table.obj2track_map.get(oid, np.empty(0, int))
        idxs = np.asarray(idxs).reshape(-1)
        idxs = idxs[idxs < gt_label.size]
        lab = gt_label[idxs]
        lab = lab[lab >= 0]
        if lab.size == 0:
            print(f"  object {oid}: no labelled tracks")
            continue
        counts = np.bincount(lab, minlength=len(gt_parts))
        dom = int(np.argmax(counts))
        purity = float(counts[dom] / counts.sum())
        comp = {gt_parts[j]: int(counts[j]) for j in range(len(gt_parts)) if counts[j]}
        print(f"  object {oid}: {int(counts.sum()):4d} tracks   dominant {gt_parts[dom]:6s} "
              f"purity {purity*100:5.1f}%   {comp}")
        scoring.append({"object": oid, "n_tracks": int(counts.sum()),
                        "dominant_gt": gt_parts[dom], "purity": purity,
                        "composition": comp})

    # how well does the decomposition cover the GT parts?
    covered = {r["dominant_gt"] for r in scoring}
    print(f"[run] GT parts covered: {len(covered)}/{len(gt_parts)} {sorted(covered)}   "
          f"objects={len(pipeline.objects)} (over-segmentation "
          f"{len(pipeline.objects) - len(gt_parts):+d})")
    mean_purity = float(np.mean([r["purity"] for r in scoring])) if scoring else 0.0
    print(f"[run] mean purity {mean_purity*100:.1f}%")

    # what did the detector actually see, before any stability gating?
    sl0 = pipeline.part_discovery.stats_log
    seen_dom = {}
    for r in sl0:
        for mem in r.get("cluster_members", []):
            m = np.asarray(mem, dtype=int)
            m = m[m < gt_label.size]
            lab = gt_label[m]
            lab = lab[lab >= 0]
            if lab.size < 5:
                continue
            cnt = np.bincount(lab, minlength=len(gt_parts))
            dom = gt_parts[int(np.argmax(cnt))]
            pur = cnt.max() / cnt.sum()
            d = seen_dom.setdefault(dom, {"n": 0, "pur": [], "size": []})
            d["n"] += 1; d["pur"].append(pur); d["size"].append(int(cnt.sum()))
    if seen_dom:
        print("[run] clusters the detector SAW (before stability gating):")
        for k_, v in sorted(seen_dom.items()):
            print(f"   dominant {k_:5s}: {v['n']:3d} clusters   "
                  f"mean purity {100*np.mean(v['pur']):5.1f}%   "
                  f"mean size {np.mean(v['size']):5.1f}")

    sl = pipeline.part_discovery.stats_log
    if sl:
        nc = np.array([r.get("n_clusters", 0) for r in sl])
        print(f"[run] clusters/update: mean {nc.mean():.2f}  max {nc.max()}  "
              f"updates with >=2: {(nc >= 2).sum()}/{len(nc)}")
        if "n_moving" in sl[-1] and "n_static" in sl[-1]:
            mv = np.array([r.get("n_moving", 0) for r in sl])
            st = np.array([r.get("n_static", 0) for r in sl])
            nz = np.array([r.get("n_noise", 0) for r in sl])
            tr = np.array([r.get("n_tracks", 0) for r in sl])
            print(f"[run] trajectories/update: tracks {tr.mean():.0f}  "
                  f"static {st.mean():.1f}  moving {mv.mean():.1f} (max {mv.max()})  "
                  f"noise {nz.mean():.1f}")

    json.dump({"sequence": reader.get_video_name(), "gt_parts": gt_parts,
               "scoring": scoring, "mean_purity": mean_purity,
               "n_objects": len(pipeline.objects),
               "gt_covered": sorted(covered),
               "cluster_stats": sl,
               "spawns": pipeline.part_spawn_log, "part_of": pipeline.part_of,
               "history": history},
              open(os.path.splitext(args.out)[0] + "_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()

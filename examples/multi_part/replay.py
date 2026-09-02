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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--object-masks", default=None)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", default="debug/multi-parts/runs/stream/replay.mp4")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--hyp-every", type=int, default=None)
    ap.add_argument("--tapir-res", type=int, default=480)
    ap.add_argument("--co-sample", type=int, default=None)
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument("--regroup-every", type=int, default=None)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    args = ap.parse_args()

    r = open_sequence(args.seq_dir)
    ext = None
    if args.object_masks:
        d = np.load(args.object_masks, allow_pickle=True)
        ext = {int(f): (d["masks"][k].max(axis=0) > 0).astype(np.uint8)
               for k, f in enumerate(d["frames"]) if k < len(d["masks"])}

    def object_mask(i):
        if ext is not None:
            return ext.get(i)
        m = np.zeros((r.H, r.W), np.uint8)
        for x in r.get_masks(i):
            m |= x
        return m

    frames = list(range(0, len(r), args.stride))
    if ext is not None:
        frames = [i for i in frames if ext.get(i) is not None]
    if args.max_frames:
        frames = frames[:args.max_frames]

    cfg = Config()
    if args.n_points:
        cfg.n_points = args.n_points
    if args.hyp_every:
        cfg.hyp_every = args.hyp_every
    if args.co_sample:
        cfg.co_sample = args.co_sample
    if args.min_frames:
        cfg.min_frames_before_split = args.min_frames
    if args.regroup_every:
        cfg.regroup_every = args.regroup_every
    tracker = TapirTracker({"checkpoint_path": args.checkpoint,
                            "resize_height": args.tapir_res, "resize_width": args.tapir_res,
                            "visible_threshold": 0.5, "device": "cuda"})
    reg = SVDClusterRANSACRegister({
        "ransac_iters": cfg.ransac_iters, "sample_size": 4,
        "inlier_thres": cfg.inlier_thres, "min_inliers": cfg.min_inliers,
        "max_clusters": cfg.max_hyp, "use_uncertainty": False})

    s = StreamingPartDiscovery(r.K, cfg, tracker, reg)
    a = frames[0]
    s.start(r.get_color(a), r.get_depth(a), object_mask(a))
    print(f"[replay] {len(s.cloud)} gaussians, stride {s.gauss_stride}, "
          f"{len(frames)} frames")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer, times, breakdown, split_costs = None, [], {}, []
    for i in frames[1:]:
        rgb, dep, m = r.get_color(i), r.get_depth(i), object_mask(i)
        if m is None:
            continue
        s.step(rgb, dep, m)
        times.append(s.last_timings["total_ms"])
        if getattr(s, "last_split_ms", None):
            split_costs.append(s.last_split_ms); s.last_split_ms = None
        for k, v in s.last_timings.items():
            breakdown.setdefault(k, []).append(v)

        vis = s.render(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy())
        bar = np.full((30, vis.shape[1], 3), (28, 24, 20), np.uint8)
        fps = 1000.0 / max(np.median(times[-30:]), 1e-6)
        label = (f"{s.state}  parts {len(s.parts)}  "
                 f"{s.last_timings['total_ms']:.0f} ms  ~{fps:.1f} fps")
        cv2.putText(bar, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
        canvas = np.vstack([vis, bar])
        if writer is None:
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     15, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    if writer:
        writer.release()

    t = np.array(times)
    print(f"[replay] wrote {args.out}")
    print(f"[replay] per-frame {np.median(t):.1f} ms median, "
          f"{np.percentile(t, 90):.1f} ms p90  ->  {1000/np.median(t):.1f} fps")
    for k, v in breakdown.items():
        print(f"[replay]   {k:12s} median {np.median(v):7.1f} ms")
    if split_costs:
        print(f"[replay]   split attempt  median {np.median(split_costs):7.1f} ms "
              f"({len(split_costs)} attempts)")
    print(f"[replay] final state: {s.state}, {len(s.parts)} parts")
    for j, p in enumerate(s.parts):
        jm = p.joint
        kind = jm.kind if jm is not None else None
        print(f"           part {j}: {int((p.weights > 0.5).sum())} gaussians, "
              f"energy {p.energy:.4f}, joint {kind}")


if __name__ == "__main__":
    main()

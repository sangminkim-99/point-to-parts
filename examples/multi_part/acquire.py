"""Watch an unknown object become controllable.

Reports, frame by frame, what the system knows about the object's joints and how
sure it is, and declares the object controllable only when that confidence has
held. The headline number is when that happens, not how fast a frame runs.

    python -m examples.multi_part.acquire --seq-dir <sequence> --view 1
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np


# what to ask the person for, when the joint is not yet pinned down
ADVICE = {
    "excitation": "move it further",
    "axis": "move it again, more slowly",
    "type": "keep moving -- the joint type is still ambiguous",
    "smooth": "hold the object steadier",
    "none": "move one part of it",
}


def limiting_factor(c):
    """Which of the three things a joint can lack is the one holding it back."""
    if not c or not c.get("n"):
        return "none"
    terms = {"excitation": c["excitation"], "type": c["type_p"],
             "smooth": c.get("smooth", 1.0)}
    a = c.get("axis_std_deg")
    terms["axis"] = 1.0 if a != a else float(np.exp(-a / 8.0))
    return min(terms, key=terms.get)


class Acquisition:
    """Confidence over time, and a conservative moment of readiness."""

    def __init__(self, thresh=0.6, hold=8):
        self.thresh, self.hold = thresh, hold
        self.rows = []
        self.streak = 0
        self.ready_at = None
        self.first_split = None

    def step(self, i, stream, t_ms):
        best, kind, lim = 0.0, None, "none"
        for p in stream.parts:
            if p.joint is None or p.joint.kind is None:
                continue
            c = p.joint.confidence()
            if c["conf"] >= best:
                best, kind, lim = c["conf"], p.joint.kind, limiting_factor(c)
        if self.first_split is None and len(stream.parts) > 1:
            self.first_split = i
        self.streak = self.streak + 1 if best >= self.thresh else 0
        if self.ready_at is None and self.streak >= self.hold:
            # date it from the frame the run of confidence began, not its end
            self.ready_at = i - self.hold + 1
        self.rows.append({"frame": i, "parts": len(stream.parts), "conf": best,
                          "kind": kind, "limit": lim, "ms": t_ms})
        return best, kind, lim

    def banner(self, stream):
        r = self.rows[-1] if self.rows else None
        if r is None:
            return "starting", (150, 150, 150)
        if self.ready_at is not None:
            return (f"READY  {r['kind']}  {100 * r['conf']:.0f}%", (90, 200, 120))
        if r["parts"] < 2:
            return ("one rigid body so far -- " + ADVICE["none"], (150, 150, 150))
        return (f"{r['kind'] or 'joint'} {100 * r['conf']:.0f}%  "
                f"({ADVICE[r['limit']]})", (70, 180, 235))

    def report(self, proc_fps, cam_fps):
        """Two clocks, and they are not the same.

        The object moves at the camera's rate, so acquisition costs a certain
        amount of OBJECT MOTION -- frames of the sequence. What it costs in wall
        clock also depends on how fast we process, and processing slower than
        the camera means seeing fewer samples of the same motion.
        """
        if not self.rows:
            return
        print()
        if self.ready_at is None:
            best = max(r["conf"] for r in self.rows)
            print(f"[acquire] never became controllable "
                  f"(best confidence {100 * best:.0f}%, needed "
                  f"{100 * self.thresh:.0f}% held for {self.hold} frames)")
        else:
            r = next(x for x in self.rows if x["frame"] >= self.ready_at)
            print(f"[acquire] CONTROLLABLE after {self.ready_at} frames of "
                  f"object motion = {self.ready_at / max(cam_fps, 1e-6):.1f} s "
                  f"at {cam_fps:.0f} Hz   ({r['kind']})")
            print(f"[acquire]   wall clock at our {proc_fps:.1f} fps: "
                  f"{self.ready_at / max(proc_fps, 1e-6):.1f} s")
            if self.first_split is not None:
                print(f"[acquire]   split at frame {self.first_split}, "
                      f"confidence then held {self.hold} frames from "
                      f"{self.ready_at}")
        # where the confidence spent its time, so a slow acquisition is legible
        lim = {}
        for r in self.rows:
            if r["parts"] > 1 and r["conf"] < self.thresh:
                lim[r["limit"]] = lim.get(r["limit"], 0) + 1
        if lim:
            print("[acquire]   time spent short of confidence: " + ", ".join(
                f"{k} {v} frames" for k, v in
                sorted(lim.items(), key=lambda x: -x[1])))

    def curve(self, path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            w.writeheader()
            w.writerows(self.rows)
        print(f"[acquire] wrote {path}")


def main():
    from point2pose.modules.tracker.tapir_tracker import TapirTracker
    from point2pose.modules.register.svd_cluster_ransac_register import (
        SVDClusterRANSACRegister)
    from examples.multi_part.naive import NaiveConfig, NaivePartTracker

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--sampler", default="uniform_fps")
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--hold", type=int, default=8,
                    help="frames the confidence must hold before READY")
    ap.add_argument("--view", type=int, default=1)
    ap.add_argument("--out", default=None, help="mp4 of the acquisition")
    ap.add_argument("--csv", default=None, help="confidence curve")
    ap.add_argument("--cam-fps", type=float, default=30.0,
                    help="the sequence's own frame rate, for the time metric")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    args = ap.parse_args()

    sd = Path(args.seq_dir).expanduser()
    if (sd / "cam_K.txt").exists():
        from examples.multi_part.recording import Recording
        r, rec = Recording(str(sd)), True
    elif (sd / "camera_rgb").is_dir():
        from point2pose.io.sources.dataset.rbo_reader import RBOReader
        r, rec = RBOReader(str(sd)), False
    else:
        from point2pose.io.sources.dataset.sapien_reader import open_sequence
        r, rec = open_sequence(str(sd)), False

    def mask_of(i):
        if rec:
            return r.get_mask(i)
        m = np.zeros((r.H, r.W), np.uint8)
        for x in r.get_masks(i):
            m |= x
        return m

    frames = list(range(0, len(r), args.stride))
    if rec:
        frames = [i for i in frames if mask_of(i) is not None]
    if args.max_frames:
        frames = frames[:args.max_frames]

    cfg = NaiveConfig(n_points=args.n_points, sampler=args.sampler)
    tracker = TapirTracker({"checkpoint_path": args.checkpoint,
                            "resize_height": cfg.tapir_res,
                            "resize_width": cfg.tapir_res,
                            "num_pips_iter": cfg.num_pips_iter,
                            "visible_threshold": 0.5, "device": "cuda"})
    reg = SVDClusterRANSACRegister({
        "ransac_iters": cfg.ransac_iters, "sample_size": 4,
        "inlier_thres": cfg.inlier_thres, "min_inliers": cfg.min_inliers,
        "max_clusters": 4, "use_uncertainty": False})

    s = NaivePartTracker(r.K, cfg, tracker, reg)
    a0 = frames[0]
    s.start(r.get_color(a0), r.get_depth(a0), mask_of(a0))
    acq = Acquisition(thresh=args.thresh, hold=args.hold)
    print(f"[acquire] {len(s.anchor_xyz)} points, {len(frames)} frames, "
          f"ready at {100 * args.thresh:.0f}% held for {args.hold} frames")

    writer, times = None, []
    if args.view:
        cv2.namedWindow("acquire", cv2.WINDOW_AUTOSIZE)
    for i in frames[1:]:
        rgb, dep, m = r.get_color(i), r.get_depth(i), mask_of(i)
        if m is None:
            continue
        t0 = time.perf_counter()
        s.step(rgb, dep, m)
        dt = (time.perf_counter() - t0) * 1e3
        times.append(dt)
        conf, kind, lim = acq.step(i, s, dt)

        if args.view or args.out:
            vis = s.render(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy())
            bar = np.full((72, vis.shape[1], 3), (26, 30, 33), np.uint8)
            txt, col = acq.banner(s)
            cv2.putText(bar, txt, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                        col, 2, cv2.LINE_AA)
            # the confidence itself, as a bar, with the readiness line on it
            w = int((vis.shape[1] - 24) * min(conf, 1.0))
            cv2.rectangle(bar, (12, 46), (vis.shape[1] - 12, 62), (58, 64, 68), -1)
            if w > 0:
                cv2.rectangle(bar, (12, 46), (12 + w, 62), col, -1)
            xt = 12 + int((vis.shape[1] - 24) * args.thresh)
            cv2.line(bar, (xt, 42), (xt, 66), (220, 220, 220), 1)
            cv2.putText(bar, f"{len(s.parts)} parts  f{i}",
                        (vis.shape[1] - 190, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (170, 170, 170), 1, cv2.LINE_AA)
            vis = np.vstack([vis, bar])
            if args.out:
                if writer is None:
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.out, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                        (vis.shape[1], vis.shape[0]))
                writer.write(vis)
            if args.view:
                cv2.imshow("acquire", vis)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord(" "):
                    while (cv2.waitKey(30) & 0xFF) != ord(" "):
                        pass
    if writer is not None:
        writer.release()
        print(f"[acquire] wrote {args.out}")
    cv2.destroyAllWindows()
    fps = 1000.0 / max(float(np.median(times)), 1e-6) if times else 0.0
    print(f"[acquire] processing {fps:.1f} fps median, "
          f"sequence assumed {args.cam_fps:.0f} Hz")
    acq.report(fps, args.cam_fps / max(args.stride, 1))
    if args.csv:
        acq.curve(args.csv)


if __name__ == "__main__":
    main()

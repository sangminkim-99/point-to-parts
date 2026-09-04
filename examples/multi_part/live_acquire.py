"""Wiggle an unknown object in front of the camera until it becomes controllable.

Click the object, press s, then move one of its parts. The bar is how sure the
system is about the joint it has found; the line on it is the threshold. When
the confidence has HELD there, the object is ready to be commanded.

    Left click   a point on the object
    Right click  a point that is not the object
    s            start
    d            depth window on/off
    r            start over
    q            quit
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from examples.multi_part.acquire import Acquisition, ADVICE


class LiveAcquire:
    """RealSense in, a controllable joint out."""

    def __init__(self, args):
        import pyrealsense2 as rs
        self.args, self.rs = args, rs
        self._init_camera(args.serial)
        self.points, self.labels = [], []
        self.started, self.sam, self.stream = False, None, None
        self.preview_mask, self._dirty = None, False
        self.acq = Acquisition(thresh=args.thresh, hold=args.hold)
        self.times = []
        self.show_depth = bool(args.depth)
        self._depth_open = False
        cv2.startWindowThread()
        cv2.namedWindow("acquire", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("acquire", self._on_mouse)
        print(__doc__)

    def _init_camera(self, serial):
        rs = self.rs
        self.pipe = rs.pipeline()
        c = rs.config()
        if serial:
            c.enable_device(str(serial))
        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        pr = self.pipe.start(c)
        it = pr.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[it.fx, 0, it.ppx], [0, it.fy, it.ppy], [0, 0, 1]],
                          dtype=np.float64)
        self.dscale = pr.get_device().first_depth_sensor().get_depth_scale()
        print(f"camera {it.width}x{it.height} fx={it.fx:.1f}")

    def _frames(self):
        f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return None, None
        return (np.asanyarray(c.get_data()),
                np.asanyarray(d.get_data()).astype(np.float32) * self.dscale)

    def _on_mouse(self, ev, x, y, *_):
        if self.started:
            return
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y]); self.labels.append(1); self._dirty = True
        elif ev == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y]); self.labels.append(0); self._dirty = True

    @staticmethod
    def _flat(logits, shape):
        if logits is None:
            return None
        m = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits)
        m = m[:, 0] if m.ndim == 4 else (m[None] if m.ndim == 2 else m)
        m = (m > 0.0).any(axis=0).astype(np.uint8)
        return (m if m.shape == shape else
                cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST))

    def _init_sam(self):
        from point2pose.modules.segmenter.sam2_real_time_segmenter import (
            Sam2RealTimeSegmenter)
        self.sam = Sam2RealTimeSegmenter({"model_cfg": self.args.sam_config,
                                          "checkpoint": self.args.sam_checkpoint,
                                          "device": "cuda"})

    # ------------------------------------------------------------------ #
    def _depth_view(self, depth, mask):
        """Depth as the tracker sees it, with the mask that survived on top.

        Coloured over the range of the masked object rather than the whole
        scene, because a few metres of background flattens exactly the
        millimetres that decide whether a part has moved.
        """
        m = None if mask is None else (mask > 0)
        v = depth[m & (depth > 0)] if m is not None and m.any() else \
            depth[depth > 0]
        if v.size:
            lo, hi = float(np.percentile(v, 2)), float(np.percentile(v, 98))
            pad = max(0.05 * (hi - lo), 0.02)
            lo, hi = lo - pad, hi + pad
        else:
            lo, hi = 0.3, 2.0
        d = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
        img = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        img[depth <= 0] = (28, 28, 28)          # no measurement at all
        if m is not None and m.any():
            e = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT,
                                 np.ones((3, 3), np.uint8))
            img[e > 0] = (255, 255, 255)
            holes = int((m & (depth <= 0)).sum())
            span = (float(v.max() - v.min()) if v.size else 0.0)
            cv2.putText(img, f"{lo:.2f}-{hi:.2f} m   object spans "
                             f"{100 * span:.0f} cm   {holes} px without depth",
                        (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def _panel(self, vis, conf, kind, i):
        """The bar, the threshold and what the person should do about it."""
        h = 96
        bar = np.full((h, vis.shape[1], 3), (26, 30, 33), np.uint8)
        txt, col = self.acq.banner(self.stream)
        cv2.putText(bar, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, col, 2,
                    cv2.LINE_AA)
        x0, x1 = 14, vis.shape[1] - 14
        cv2.rectangle(bar, (x0, 52), (x1, 72), (58, 64, 68), -1)
        w = int((x1 - x0) * float(np.clip(conf, 0, 1)))
        if w:
            cv2.rectangle(bar, (x0, 52), (x0 + w, 72), col, -1)
        xt = x0 + int((x1 - x0) * self.args.thresh)
        cv2.line(bar, (xt, 47), (xt, 77), (225, 225, 225), 2)
        # the streak is the conservative part: the bar has to STAY past the line
        cv2.putText(bar, f"held {self.acq.streak}/{self.args.hold}",
                    (x0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (170, 170, 170), 1, cv2.LINE_AA)
        parts = 0 if self.stream is None else len(self.stream.parts)
        fps = 1000.0 / max(float(np.median(self.times[-30:])), 1e-6) \
            if self.times else 0.0
        cv2.putText(bar, f"{parts} parts   f{i}   {fps:.0f} fps",
                    (x1 - 230, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (170, 170, 170), 1, cv2.LINE_AA)
        return np.vstack([vis, bar])

    def _readout(self, vis):
        """Per-joint numbers, so a stuck acquisition is legible, not mysterious."""
        y = 24
        for j, p in enumerate(self.stream.parts):
            if p.joint is None or p.joint.kind is None:
                continue
            c = p.joint.confidence()
            span = (np.degrees(c["span"]) if p.joint.kind == "revolute"
                    else c["span"] * 1000)
            unit = "deg" if p.joint.kind == "revolute" else "mm"
            cv2.putText(vis, f"p{j} {p.joint.kind[:5]} {100*c['conf']:3.0f}%  "
                             f"type {c['type_p']:.2f}  axis +-"
                             f"{c['axis_std_deg']:.1f}deg  moved {span:.0f}{unit}"
                             f"  smooth {c.get('smooth', 0):.2f}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (235, 235, 235), 1, cv2.LINE_AA)
            y += 18
            # what the type decision actually came down to, and on what noise
            b = c.get("bic", {})
            best = min(b.values()) if b else 0.0
            marg = "  ".join(f"{k[:4]} +{v - best:.0f}" for k, v in
                             sorted(b.items(), key=lambda x: x[1]))
            cv2.putText(vis, f"     BIC {marg}   noise {1000*c.get('sigma_t',0):.1f}mm"
                             f" / {np.degrees(c.get('sigma_r', 0)):.1f}deg"
                             f"   rmse {c.get('rmse', 0):.1f}",
                        (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (170, 175, 180), 1, cv2.LINE_AA)
            y += 22
        return vis

    def run(self):
        from point2pose.modules.tracker.tapir_tracker import TapirTracker
        from point2pose.modules.register.svd_cluster_ransac_register import (
            SVDClusterRANSACRegister)
        from examples.multi_part.naive import NaiveConfig, NaivePartTracker
        from examples.multi_part.streaming import clean_mask

        cfg = NaiveConfig(n_points=self.args.n_points, sampler=self.args.sampler)
        for k, v in (("pending", self.args.pending),
                     ("merge_rigid", self.args.merge_rigid),
                     ("reproject", self.args.reproject)):
            if v is not None:
                setattr(cfg, k, bool(v))
        tracker = reg = None
        i = 0
        try:
            while True:
                bgr, depth = self._frames()
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                disp = bgr.copy()

                if not self.started:
                    if self._dirty and self.points:
                        if self.sam is None:
                            self._init_sam()
                        try:
                            self.preview_mask = self._flat(
                                self.sam.preview(rgb, [self.points],
                                                 [self.labels]), rgb.shape[:2])

                        except Exception as e:
                            print(f"[sam] {e}")
                        self._dirty = False
                    if self.preview_mask is not None:
                        ov = np.zeros_like(disp)
                        ov[self.preview_mask > 0] = (60, 200, 60)
                        disp = cv2.addWeighted(disp, 1.0, ov, 0.4, 0)
                        # what the tracker will actually use, after depth cleaning
                        cm = clean_mask(self.preview_mask, depth) > 0
                        e = cv2.morphologyEx(cm.astype(np.uint8),
                                             cv2.MORPH_GRADIENT,
                                             np.ones((3, 3), np.uint8))
                        disp[e > 0] = (235, 235, 235)
                    for (x, y), l in zip(self.points, self.labels):
                        cv2.circle(disp, (x, y), 5,
                                   (60, 220, 60) if l else (60, 60, 235), -1)
                    cv2.putText(disp, "click the object, then press s",
                                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (235, 235, 235), 2, cv2.LINE_AA)
                    disp = self._panel(disp, 0.0, None, i)
                else:
                    _, logits = self.sam.segment(rgb)
                    mask = self._flat(logits, rgb.shape[:2])
                    if mask is None or mask.sum() < 200:
                        cv2.putText(disp, "mask lost", (12, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (60, 60, 235), 2, cv2.LINE_AA)
                        disp = self._panel(disp, 0.0, None, i)
                    else:
                        t0 = time.perf_counter()
                        self.stream.step(rgb, depth, mask)
                        dt = (time.perf_counter() - t0) * 1e3
                        self.times.append(dt)
                        i += 1
                        conf, kind, _ = self.acq.step(i, self.stream, dt)
                        disp = self.stream.render(disp)
                        disp = self._readout(disp)
                        disp = self._panel(disp, conf, kind, i)

                if self.show_depth:
                    shown = (getattr(self.stream, "mask", None)
                             if self.started else
                             (None if self.preview_mask is None else
                              clean_mask(self.preview_mask, depth)))
                    cv2.imshow("depth", self._depth_view(depth, shown))
                elif self._depth_open:
                    cv2.destroyWindow("depth")
                self._depth_open = self.show_depth
                cv2.imshow("acquire", disp)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("d"):
                    self.show_depth = not self.show_depth
                if k == ord("r"):
                    self.points, self.labels = [], []
                    self.started, self.stream = False, None
                    self.preview_mask = None
                    self.acq = Acquisition(thresh=self.args.thresh,
                                           hold=self.args.hold)
                    self.times, i = [], 0
                    print("reset")
                if k == ord("s") and not self.started and self.points:
                    if self.sam is None:
                        self._init_sam()
                    self.sam.add_input_object(self.points, self.labels)
                    self.sam.initialize(rgb)
                    _, logits = self.sam.segment(rgb)
                    mask = self._flat(logits, rgb.shape[:2])
                    if mask is None or mask.sum() < 200:
                        print("mask too small; click again")
                        continue
                    if tracker is None:
                        tracker = TapirTracker({
                            "checkpoint_path": self.args.checkpoint,
                            "resize_height": cfg.tapir_res,
                            "resize_width": cfg.tapir_res,
                            "num_pips_iter": cfg.num_pips_iter,
                            "visible_threshold": 0.5, "device": "cuda"})
                        reg = SVDClusterRANSACRegister({
                            "ransac_iters": cfg.ransac_iters, "sample_size": 4,
                            "inlier_thres": cfg.inlier_thres,
                            "min_inliers": cfg.min_inliers,
                            "max_clusters": 4, "use_uncertainty": False})
                    self.stream = NaivePartTracker(self.K, cfg, tracker, reg)
                    self.stream.start(rgb, depth, mask)
                    self.started = True
                    print(f"started with {len(self.stream.anchor_xyz)} points "
                          f"-- now move one part of the object")
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()
            fps = 1000.0 / max(float(np.median(self.times)), 1e-6) \
                if self.times else 0.0
            # live: the two clocks are the same, we see what we process
            self.acq.report(fps, fps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None)
    ap.add_argument("--n-points", type=int, default=200)
    ap.add_argument("--sampler", default="uniform_fps")
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--depth", type=int, default=0,
                    help="open the depth window at startup ('d' toggles it)")
    ap.add_argument("--pending", type=int, default=None)
    ap.add_argument("--merge-rigid", type=int, default=None)
    ap.add_argument("--reproject", type=int, default=None)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--sam-checkpoint",
                    default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    LiveAcquire(ap.parse_args()).run()


if __name__ == "__main__":
    main()

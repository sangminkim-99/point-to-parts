"""Live multi-part discovery and per-part 6-DoF tracking from a RealSense stream.

Click the object once on the first frame; SAM 2 turns the clicks into a mask and
propagates it. The object then starts as one rigid body with a single box. As it
articulates, the parts separate on their own -- no CAD model, no part count, no
category prior -- and each one gets its own dense model, oriented box, 6-DoF pose
axes and joint axis.

    Left click   add a positive point
    Right click  add a negative point
    s            start
    h            toggle the per-hypothesis residual strip
    r            reset
    q            quit

Front-end modelled on `examples/realsense_tracking/realsense_tracking.py`; the
method itself lives in `streaming.py`.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np

from examples.multi_part.streaming import Config, StreamingPartDiscovery


class RealSenseMultiPart:
    def __init__(self, args):
        import pyrealsense2 as rs

        self.args = args
        self.cfg = Config()
        if args.n_points:
            self.cfg.n_points = args.n_points
        if args.hyp_every:
            self.cfg.hyp_every = args.hyp_every
        if args.pips_iter:
            self.cfg.num_pips_iter = args.pips_iter

        self.rs = rs
        self._init_realsense(args.serial)

        self.points, self.labels = [], []
        self.started = False
        self.preview_mask = None
        self._preview_dirty = False
        self.stream = None
        self.sam = None
        self.times = []
        self.show_hyp = True

        cv2.startWindowThread()
        cv2.namedWindow("multi-part", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("multi-part", self._on_mouse)
        print(__doc__)

    # ------------------------------------------------------------------ #
    def _init_realsense(self, serial):
        rs = self.rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.align = rs.align(rs.stream.color)
        profile = self.pipe.start(cfg)
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.K = np.array([[intr.fx, 0, intr.ppx],
                           [0, intr.fy, intr.ppy],
                           [0, 0, 1]], dtype=np.float64)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"camera {intr.width}x{intr.height}  fx={intr.fx:.1f}  "
              f"depth scale {self.depth_scale}")

    def _frames(self):
        f = self.align.process(self.pipe.wait_for_frames())
        c, d = f.get_color_frame(), f.get_depth_frame()
        if not c or not d:
            return None, None
        bgr = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale
        return bgr, depth

    def _on_mouse(self, event, x, y, _flags, _param):
        if self.started:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y]); self.labels.append(1)
            self._preview_dirty = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.points.append([x, y]); self.labels.append(0)
            self._preview_dirty = True

    # ------------------------------------------------------------------ #
    def _init_sam(self):
        """SAM 2 real-time: prompt once on the first frame, then track."""
        from point2pose.modules.segmenter.sam2_real_time_segmenter import (
            Sam2RealTimeSegmenter)
        self.sam = Sam2RealTimeSegmenter({
            "model_cfg": self.args.sam_config,
            "checkpoint": self.args.sam_checkpoint,
            "device": "cuda"})

    @staticmethod
    def _flatten(logits, shape):
        """SAM2 returns per-object logits [N,1,H,W]; the streaming method wants one
        binary mask of the WHOLE object, parts unknown."""
        if logits is None:
            return None
        m = np.asarray(logits.detach().cpu() if hasattr(logits, "detach") else logits)
        if m.ndim == 4:
            m = m[:, 0]
        elif m.ndim == 2:
            m = m[None]
        m = (m > 0.0).any(axis=0).astype(np.uint8)
        if m.shape != shape:
            m = cv2.resize(m, (shape[1], shape[0]),
                           interpolation=cv2.INTER_NEAREST)
        return m

    def _preview(self, rgb):
        masks = self.sam.preview(rgb, [self.points], [self.labels])
        return self._flatten(masks, rgb.shape[:2])

    # ------------------------------------------------------------------ #
    def run(self):
        from point2pose.modules.register.svd_cluster_ransac_register import (
            SVDClusterRANSACRegister)
        from point2pose.modules.tracker.tapir_tracker import TapirTracker

        tracker = reg = None
        try:
            while True:
                bgr, depth = self._frames()
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                disp = bgr.copy()

                if not self.started:
                    if self._preview_dirty and self.points:
                        if self.sam is None:
                            self._init_sam()
                        try:
                            self.preview_mask = self._preview(rgb)
                        except Exception as e:
                            print(f"[sam] preview failed: {e}")
                            self.preview_mask = None
                        self._preview_dirty = False
                    if self.preview_mask is not None:
                        ov = np.zeros_like(disp)
                        ov[self.preview_mask > 0] = (60, 200, 60)
                        disp = cv2.addWeighted(disp, 1.0, ov, 0.45, 0)
                    for (x, y), l in zip(self.points, self.labels):
                        if l == 1:
                            cv2.circle(disp, (x, y), 5, (60, 220, 60), -1)
                        else:
                            cv2.drawMarker(disp, (x, y), (0, 0, 255),
                                           cv2.MARKER_TILTED_CROSS, 12, 2)
                    cv2.putText(disp, "L +point  R -point   's' start  'r' reset  'q' quit",
                                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    _, logits = self.sam.segment(rgb)
                    mask = self._flatten(logits, rgb.shape[:2])
                    if mask is None:
                        mask = np.zeros(rgb.shape[:2], np.uint8)
                    if mask.sum() < 200:
                        cv2.putText(disp, "mask lost", (10, 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else:
                        self.stream.step(rgb, depth, mask)
                        self.times.append(self.stream.last_timings["total_ms"])
                        disp = self.stream.render(disp)
                        if self.show_hyp:
                            panel = self.stream.hypothesis_panel(
                                depth, mask, width=disp.shape[1] // 4)
                            if panel is not None:
                                pad = np.full(
                                    (panel.shape[0], disp.shape[1] - panel.shape[1], 3),
                                    (32, 30, 28), np.uint8)
                                disp = np.vstack([disp, np.hstack([panel, pad])])

                    fps = 1000.0 / max(np.median(self.times[-30:]), 1e-6) if self.times else 0
                    t = self.stream.last_timings
                    bar = np.full((52, disp.shape[1], 3), (28, 24, 20), np.uint8)
                    cv2.putText(bar, f"{self.stream.state}   parts {len(self.stream.parts)}"
                                     f"   {t.get('total_ms', 0):.0f} ms   {fps:.1f} fps",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (235, 235, 235), 1, cv2.LINE_AA)
                    cv2.putText(bar, f"track {t.get('track_ms', 0):.0f}  "
                                     f"hyp {t.get('hyp_ms', 0):.0f}  "
                                     f"dense {t.get('dense_ms', 0):.0f}  (ms)",
                                (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (150, 150, 150), 1, cv2.LINE_AA)
                    disp = np.vstack([disp, bar])

                cv2.imshow("multi-part", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("h"):
                    self.show_hyp = not self.show_hyp
                if key == ord("r"):
                    self.points, self.labels = [], []
                    self.started = False
                    self.preview_mask = None
                    self.stream = None
                    self.times = []
                    print("reset")
                if key == ord("s") and not self.started:
                    if not any(l == 1 for l in self.labels):
                        print("click at least one positive point first")
                        continue
                    if self.sam is None:
                        self._init_sam()
                    self.sam.clear_input_objects()
                    self.sam.add_input_object(self.points, self.labels)
                    self.sam.initialize(rgb)
                    _, logits = self.sam.segment(rgb)
                    mask = self._flatten(logits, rgb.shape[:2])
                    if mask is None or mask.sum() < 200:
                        print("mask too small; click again")
                        continue
                    if tracker is None:
                        tracker = TapirTracker({
                            "checkpoint_path": self.args.checkpoint,
                            "resize_height": self.args.tapir_res,
                            "resize_width": self.args.tapir_res,
                            "num_pips_iter": self.cfg.num_pips_iter,
                            "visible_threshold": 0.5, "device": "cuda"})
                        reg = SVDClusterRANSACRegister({
                            "ransac_iters": self.cfg.ransac_iters, "sample_size": 4,
                            "inlier_thres": self.cfg.inlier_thres,
                            "min_inliers": self.cfg.min_inliers,
                            "max_clusters": self.cfg.max_hyp,
                            "use_uncertainty": False})
                    self.stream = StreamingPartDiscovery(self.K, self.cfg, tracker, reg)
                    self.stream.start(rgb, depth, mask)
                    self.started = True
                    print(f"started: {len(self.stream.cloud)} gaussians, "
                          f"stride {self.stream.gauss_stride}")
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="RealSense serial number")
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--hyp-every", type=int, default=None)
    ap.add_argument("--tapir-res", type=int, default=480)
    ap.add_argument("--pips-iter", type=int, default=None)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--sam-checkpoint",
                    default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    RealSenseMultiPart(ap.parse_args()).run()


if __name__ == "__main__":
    main()

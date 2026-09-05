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
    u            write the URDF (needs --urdf PATH)
    q            quit

--method naive runs the sparse-only tracker in `naive.py` (steps 1-3), with the
gaussian model and the render-based refinement as optional layers on top of it;
--method dense runs the gaussian pipeline in `streaming.py`. Front-end modelled
on `examples/realsense_tracking/realsense_tracking.py`.
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
        if getattr(args, "method", "dense") == "naive":
            from examples.multi_part.naive import NaiveConfig
            self.cfg = NaiveConfig(dense=bool(getattr(args, "dense", 0)),
                                   refine=bool(getattr(args, "dense", 0)
                                               and getattr(args, "refine", 0)))
        else:
            self.cfg = Config()
        if args.n_points:
            self.cfg.n_points = args.n_points
        for k, v in (("sampler", args.sampler),
                     ("split_out_pts", args.split_out_pts),
                     ("min_part_pts", args.min_part_pts),
                     ("joint_track", None if args.joint_track is None
                      else bool(args.joint_track)),
                     ("persist", None if args.persist is None
                      else bool(args.persist))):
            if v is not None and hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        if args.hyp_every and hasattr(self.cfg, "hyp_every"):
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

    def _write_urdf(self):
        """Step 6 on demand: meshes come from whatever the model holds now."""
        from examples.multi_part import urdf_export as ux
        st = self.stream
        model = getattr(st, "model", None)
        if model is not None:
            gm = model.cloud.means.detach().cpu().numpy()
            pts_of = lambda j: gm[model.labels[:len(gm)] == j]
        elif hasattr(st, "anchor_xyz"):
            pts_of = lambda j: st.anchor_xyz[st.parts[j].idx]
        else:
            gm = st.cloud.means.detach().cpu().numpy()
            pts_of = lambda j: gm[st.parts[j].weights[:len(gm)] > 0.5]
        try:
            out, w = ux.export(self.args.urdf, st.parts, pts_of)
            print(f"wrote {out} with {len(w)} collision meshes")
        except Exception as exc:
            print(f"urdf export failed: {exc}")

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
                        # the preview is SAM 2's raw mask; the tracker will use
                        # the depth-cleaned one, so show that outline too
                        try:
                            from examples.multi_part.streaming import clean_mask
                            cm = clean_mask(self.preview_mask, depth) > 0
                            e = cv2.morphologyEx(cm.astype(np.uint8),
                                                 cv2.MORPH_GRADIENT,
                                                 np.ones((3, 3), np.uint8))
                            disp[e > 0] = (235, 235, 235)
                        except Exception:
                            pass
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
                        if self.show_hyp and hasattr(self.stream,
                                                     "hypothesis_panel"):
                            panel = self.stream.hypothesis_panel(
                                depth, mask, width=disp.shape[1] // 4)
                            if panel is not None:
                                disp = np.vstack(
                                    [disp, self.stream.fit_panel(panel, disp.shape[1])])

                    fps = 1000.0 / max(np.median(self.times[-30:]), 1e-6) if self.times else 0
                    t = self.stream.last_timings
                    bar = np.full((86, disp.shape[1], 3), (28, 24, 20), np.uint8)
                    cv2.putText(bar, f"{self.stream.state}   parts {len(self.stream.parts)}"
                                     f"   {t.get('total_ms', 0):.0f} ms   {fps:.1f} fps",
                                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (235, 235, 235), 1, cv2.LINE_AA)
                    cv2.putText(bar, f"track {t.get('track_ms', 0):.0f}  "
                                     f"hyp {t.get('hyp_ms', 0):.0f}  "
                                     f"dense {t.get('dense_ms', 0):.0f}  (ms)",
                                (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (150, 150, 150), 1, cv2.LINE_AA)
                    d = getattr(self.stream, "diag", {})
                    if self.args.method == "naive":
                        # outliers, not the median residual, are what a split
                        # needs: the second part is always the smaller set
                        hot = any(p.out_pts >= self.cfg.split_out_pts
                                  for p in self.stream.parts)
                        cv2.putText(bar, "  ".join(
                            f"p{j} {p.resid*1000:.0f}mm out {p.out_pts}pt"
                            + (" joint" if p.on_joint else "")
                            for j, p in enumerate(self.stream.parts)),
                            (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                            (150, 220, 150) if hot else (150, 150, 150), 1,
                            cv2.LINE_AA)
                        cv2.putText(bar, f"points {len(self.stream.anchor_xyz)}"
                                         f"   need out > "
                                         f"{self.cfg.split_out_pts}pt"
                                         f" for {self.cfg.split_frames} frames",
                                    (8, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (170, 170, 170), 1, cv2.LINE_AA)
                    else:
                        # a split needs >=2 DISTINCT hypotheses; with one, every
                        # gaussian has the same posterior and nothing separates
                        hcol = (90, 90, 235) if d.get("hyp", 0) < 2 else (150, 220, 150)
                        cv2.putText(bar, f"hypotheses {d.get('hyp', 0)}   "
                                         f"tracks {d.get('tracks_live', 0)}/{d.get('tracks', 0)}"
                                         f"   decisive {100*d.get('decisive', 0):.0f}%"
                                         f"   split tries {d.get('tries', 0)}",
                                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                    hcol, 1, cv2.LINE_AA)
                        cv2.putText(bar, d.get("why", ""), (8, 76),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (170, 170, 170), 1, cv2.LINE_AA)
                    disp = np.vstack([disp, bar])

                cv2.imshow("multi-part", disp)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("u") and self.stream is not None \
                        and self.args.urdf:
                    self._write_urdf()
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
                            "max_clusters": getattr(self.cfg, "max_hyp", 4),
                            "use_uncertainty": False})
                    if self.args.method == "naive":
                        from examples.multi_part.naive import NaivePartTracker
                        self.stream = NaivePartTracker(self.K, self.cfg,
                                                       tracker, reg)
                    else:
                        self.stream = StreamingPartDiscovery(self.K, self.cfg,
                                                             tracker, reg)
                    self.stream.start(rgb, depth, mask)
                    self.started = True
                    if self.args.method == "naive":
                        print(f"started: {len(self.stream.anchor_xyz)} points"
                              + (f", {len(self.stream.model.cloud)} gaussians"
                                 if self.stream.model else ""))
                    else:
                        print(f"started: {len(self.stream.cloud)} gaussians, "
                              f"stride {self.stream.gauss_stride}")
        finally:
            self.pipe.stop()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serial", default=None, help="RealSense serial number")
    ap.add_argument("--method", choices=["dense", "naive"], default="dense",
                    help="naive = sparse frontend only (steps 1-3)")
    ap.add_argument("--dense", type=int, default=0,
                    help="naive only: add the gaussian model (step 4)")
    ap.add_argument("--refine", type=int, default=0,
                    help="naive only: add render-based refinement (step 5)")
    ap.add_argument("--urdf", default=None,
                    help="write a URDF on quit")
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--sampler", default=None,
                    help="super_point_balanced | super_point_fps | uniform_fps "
                         "| orb | random")
    ap.add_argument("--split-out-pts", type=int, default=None)
    ap.add_argument("--min-part-pts", type=int, default=None)
    ap.add_argument("--joint-track", type=int, default=None)
    ap.add_argument("--persist", type=int, default=None)
    ap.add_argument("--hyp-every", type=int, default=None)
    ap.add_argument("--tapir-res", type=int, default=512)
    ap.add_argument("--pips-iter", type=int, default=None)
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--sam-checkpoint",
                    default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    RealSenseMultiPart(ap.parse_args()).run()


if __name__ == "__main__":
    main()

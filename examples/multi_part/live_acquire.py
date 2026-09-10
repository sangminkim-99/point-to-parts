"""Observe articulation from a live RGB-D stream.

Click the object, press s, then move one of its parts. The bar is how sure the
system is about the joint it has found; the line on it is the threshold.
Stable estimates are diagnostics, not verification of robot controllability.

    Left click   a point on the object   (drag a box with --bbox-prompt)
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

from examples.multi_part.acquire import (Acquisition, ADVICE, apply_config,
                                         apply_overrides)


class LiveAcquire:
    """RealSense input and live joint-estimation diagnostics."""

    def __init__(self, args):
        import pyrealsense2 as rs
        self.args, self.rs = args, rs
        self._init_camera(args.serial)
        self.points, self.labels = [], []
        self.bbox, self._drag = None, None
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
        if self.args.bbox_prompt:
            # left drag draws the box; a right click still adds a negative point
            if ev == cv2.EVENT_LBUTTONDOWN:
                self._drag = [x, y, x, y]
            elif ev == cv2.EVENT_MOUSEMOVE and self._drag is not None:
                self._drag[2:] = [x, y]
            elif ev == cv2.EVENT_LBUTTONUP and self._drag is not None:
                x0, y0, x1, y1 = self._drag
                self._drag = None
                if abs(x1 - x0) > 8 and abs(y1 - y0) > 8:
                    self.bbox = [min(x0, x1), min(y0, y1),
                                 max(x0, x1), max(y0, y1)]
                    self._dirty = True
            elif ev == cv2.EVENT_RBUTTONDOWN:
                self.points.append([x, y]); self.labels.append(0)
                self._dirty = True
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
        """One row per joint: the gate, the confidence, and the range.

        These answer different questions -- is it a joint, which kind is it,
        where is it, how far does it go -- so they are shown apart. Only "which
        kind" and "where" are about knowing the joint, so only they make the
        bar; the gate decides whether the row is a joint at all, and the range
        is observed motion, not a verified mechanical limit.
        """
        joints = [(j, p) for j, p in enumerate(self.stream.parts)
                  if p.joint is not None and p.joint.kind] if self.stream else []
        rows = max(1, len(joints))
        bar = np.full((40 + 40 * rows, vis.shape[1], 3), (26, 30, 33), np.uint8)
        txt, col = self.acq.banner(self.stream)
        cv2.putText(bar, txt, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, col, 2,
                    cv2.LINE_AA)
        parts = 0 if self.stream is None else len(self.stream.parts)
        fps = 1000.0 / max(float(np.median(self.times[-30:])), 1e-6) \
            if self.times else 0.0
        cv2.putText(bar, f"{parts} parts   f{i}   {fps:.0f} fps   "
                         f"held {self.acq.streak}/{self.args.hold}",
                    (vis.shape[1] - 320, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                    (170, 170, 170), 1, cv2.LINE_AA)
        pal = [(60, 140, 235), (200, 120, 40), (70, 180, 90), (200, 80, 200),
               (60, 200, 200), (90, 90, 235)]
        x0, x1 = 110, vis.shape[1] - 14
        for k, (j, p) in enumerate(joints or [(None, None)]):
            y = 42 + 40 * k
            if p is None:
                cv2.putText(bar, "no joint yet", (14, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1,
                            cv2.LINE_AA)
                continue
            c = p.joint.confidence()
            v = float(np.clip(c["conf"], 0, 1))
            ok = c.get("valid", True)
            pc = pal[j % len(pal)]
            fill = ((90, 200, 120) if (ok and v >= self.args.thresh)
                    else (pc if ok else (90, 90, 95)))
            cv2.putText(bar, f"p{j} {p.joint.kind[:5]}", (14, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, pc, 1, cv2.LINE_AA)
            cv2.rectangle(bar, (x0, y + 2), (x1, y + 20), (58, 64, 68), -1)
            w = int((x1 - x0) * v)
            if w:
                cv2.rectangle(bar, (x0, y + 2), (x0 + w, y + 20), fill, -1)
            xt = x0 + int((x1 - x0) * self.args.thresh)
            cv2.line(bar, (xt, y - 1), (xt, y + 23), (225, 225, 225), 2)
            cv2.putText(bar, f"{100 * v:3.0f}%", (x0 + 6, y + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (245, 245, 245), 1,
                        cv2.LINE_AA)
            # what was watched, in units. A percentage would claim to know
            # where the joint stops, and nothing here can know that.
            lim = p.joint.limits()
            rev = p.joint.kind == "revolute"
            f = (lambda x: np.degrees(x)) if rev else (lambda x: 1000 * x)
            u = "deg" if rev else "mm"
            rng = ("" if lim is None else
                   f"{f(lim[0]):.0f} to {f(lim[1]):.0f} {u}")
            need = f(c.get("need", 0.0))
            cv2.putText(bar,
                        f"joint? {'yes' if ok else 'NO'} (smooth "
                        f"{c.get('smooth', 0):.2f})    type {c['type_p']:.2f}"
                        f"    axis RMS {c['axis_std_deg']:.1f} deg"
                        + (f"    watched {rng}" if rng else "")
                        + (f" (need {need:.0f} {u})"
                           if c["excitation"] < 1.0 else ""),
                        (x0, y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (235, 235, 235) if ok else (140, 140, 145), 1,
                        cv2.LINE_AA)
        return np.vstack([vis, bar])

    def run(self):
        from point2pose.modules.tracker.tapir_tracker import TapirTracker
        from point2pose.modules.register.svd_cluster_ransac_register import (
            SVDClusterRANSACRegister)
        from examples.multi_part.naive import NaiveConfig, NaivePartTracker
        from examples.multi_part.streaming import clean_mask

        cfg = NaiveConfig()
        apply_config(cfg, self.args.config)
        if self.args.n_points:
            cfg.n_points = self.args.n_points
        if self.args.sampler:
            cfg.sampler = self.args.sampler
        for k, v in (("pending", self.args.pending),
                     ("merge_rigid", self.args.merge_rigid),
                     ("reproject", self.args.reproject)):
            if v is not None:
                setattr(cfg, k, bool(v))
        for k, v in (("reseed_points", self.args.reseed_points),
                     ("reseed_gap", self.args.reseed_gap),
                     ("min_live", self.args.min_live),
                     ("key_angle_deg", self.args.key_angle_deg),
                     ("motion_sigma", self.args.motion_sigma),
                     ("split_min_rot_deg", self.args.split_min_rot_deg),
                     ("split_min_trans_m", self.args.split_min_trans_m),
                     ("rot_tol", self.args.rot_tol)):
            if v is not None:
                setattr(cfg, k, v)
        apply_overrides(cfg, self.args.set)
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
                    if self._dirty and (self.points or self.bbox is not None):
                        if self.sam is None:
                            self._init_sam()
                        try:
                            self.preview_mask = self._flat(
                                self.sam.preview(rgb, [self.points], [self.labels],
                                                 [self.bbox]), rgb.shape[:2])

                        except Exception as e:
                            print(f"[sam] {e}")
                        self._dirty = False
                    if self.preview_mask is not None:
                        ov = np.zeros_like(disp)
                        ov[self.preview_mask > 0] = (60, 200, 60)
                        disp = cv2.addWeighted(disp, 1.0, ov, 0.4, 0)
                        # what the tracker will actually use, after depth cleaning
                        cm = clean_mask(self.preview_mask, depth, K=self.K) > 0
                        e = cv2.morphologyEx(cm.astype(np.uint8),
                                             cv2.MORPH_GRADIENT,
                                             np.ones((3, 3), np.uint8))
                        disp[e > 0] = (235, 235, 235)
                    for (x, y), l in zip(self.points, self.labels):
                        cv2.circle(disp, (x, y), 5,
                                   (60, 220, 60) if l else (60, 60, 235), -1)
                    box = self._drag or self.bbox
                    if box is not None:
                        cv2.rectangle(disp, (int(box[0]), int(box[1])),
                                      (int(box[2]), int(box[3])),
                                      (60, 220, 60), 2)
                    cv2.putText(disp,
                                "drag a box round the object, then press s"
                                if self.args.bbox_prompt else
                                "click the object, then press s",
                                (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (235, 235, 235), 2, cv2.LINE_AA)
                    disp = self._panel(disp, 0.0, None, i)
                else:
                    _, logits = self.sam.segment(rgb)
                    mask = self._flat(logits, rgb.shape[:2])
                    if mask is None or mask.sum() < 200:
                        self.acq.invalidate_tracking()
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
                        disp = self.stream.render(disp, style=self.args.vis)
                        disp = self._panel(disp, conf, kind, i)

                if self.show_depth:
                    shown = (getattr(self.stream, "mask", None)
                             if self.started else
                             (None if self.preview_mask is None else
                              clean_mask(self.preview_mask, depth, K=self.K)))
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
                    self.bbox, self._drag = None, None
                    self.started, self.stream = False, None
                    self.preview_mask = None
                    self.acq = Acquisition(thresh=self.args.thresh,
                                           hold=self.args.hold)
                    self.times, i = [], 0
                    print("reset")
                if k == ord("s") and not self.started \
                        and (self.points or self.bbox is not None):
                    if self.sam is None:
                        self._init_sam()
                    self.sam.add_input_object(self.points, self.labels, self.bbox)
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
    ap.add_argument("--vis", choices=["clean", "debug"], default="clean",
                    help="clean shows what was found, debug shows why")
    ap.add_argument("--bbox-prompt", action="store_true",
                    help="drag a box instead of clicking points")
    ap.add_argument("--config", default=None,
                    help="a file in configs/multi-part, or a path")
    ap.add_argument("--n-points", type=int, default=None)
    ap.add_argument("--sampler", default=None)
    ap.add_argument("--thresh", type=float, default=0.6)
    ap.add_argument("--hold", type=int, default=8)
    ap.add_argument("--depth", type=int, default=0,
                    help="open the depth window at startup ('d' toggles it)")
    ap.add_argument("--pending", type=int, default=None)
    ap.add_argument("--merge-rigid", type=int, default=None)
    ap.add_argument("--reproject", type=int, default=None)
    ap.add_argument("--reseed-points", type=int, default=None)
    ap.add_argument("--reseed-gap", type=int, default=None)
    ap.add_argument("--min-live", type=int, default=None)
    ap.add_argument("--key-angle-deg", type=float, default=None)
    ap.add_argument("--motion-sigma", type=float, default=None)
    ap.add_argument("--split-min-rot-deg", type=float, default=None)
    ap.add_argument("--split-min-trans-m", type=float, default=None)
    ap.add_argument("--rot-tol", type=float, default=None)
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="override any NaiveConfig field, e.g. --set "
                         "split_out_pts=4 --set co_gap=0.3")
    ap.add_argument("--checkpoint",
                    default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--sam-checkpoint",
                    default="checkpoints/sam2.1/sam2.1_hiera_large.pt")
    ap.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    LiveAcquire(ap.parse_args()).run()


if __name__ == "__main__":
    main()

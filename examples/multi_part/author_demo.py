"""Interactive point2pose model authoring: live RealSense or causal RGB-D replay.

python -m examples.multi_part.author_demo --config reprojection_split.yaml --bbox-prompt
python -m examples.multi_part.author_demo --seq-dir <recording> --config reprojection_split.yaml
"""
import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import threading
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from examples.multi_part.naive import NaivePartTracker, NaiveConfig
from examples.multi_part.acquire import apply_config, apply_overrides
from examples.multi_part.author_state import (snapshot, displayed_poses, save_snapshot,
                                               cloud_center_radius, frame_view,
                                               PALETTE, geometry_counts, legend_markdown,
                                               render_status_markdown, uncertainty_markdown)


class ReferenceTracker(NaivePartTracker):
    """Keep a chosen identity as reference; its SE(3) pose is still tracked."""
    reference_id = 0

    def start(self, rgb, depth, mask):
        super().start(rgb, depth, mask)
        self.root = 0
        if getattr(self, 'recorder', None):
            self.recorder(rgb, depth, mask)
        return self

    def step(self, rgb, depth, mask):
        result = super().step(rgb, depth, mask)
        if getattr(self, 'recorder', None):
            self.recorder(rgb, depth, mask)
        return result

    def _pick_root(self):
        for j, p in enumerate(self.parts):
            if p.part_id == self.reference_id:
                return j
        raise RuntimeError('Selected reference identity disappeared; reset or select it again')

    def select_reference(self, part_id):
        if part_id not in [p.part_id for p in self.parts]:
            raise ValueError('Reference part no longer exists')
        self.reference_id = int(part_id)
        self._reparent()


class AuthorView:
    def __init__(self, args):
        import viser
        self.args = args
        self.lock = threading.RLock()
        self.commands = deque()
        self.state = self.frozen = None
        self.nodes = []
        self.events = []
        self.last_update = 0.
        self.source = getattr(args, 'source', 'live')   # 'live' or 'replay'
        self.center = None            # display-frame cloud centroid, for framing
        self.radius = 0.
        self.framed = False           # have we auto-framed once data arrived?
        self.counts = None            # assigned/unassigned geometry counts
        # Diagnostics: default off + low-rate so overhead stays near zero. The
        # actual Gaussian render is produced in the tracking thread (publish),
        # never in a GUI callback, and is snapshotted for the GUI to show.
        self.diag_rgb = self.diag_resid = self.diag_obs = None
        self.diag_coverage = self.diag_agreement = None
        self.last_render_frame = None
        self.last_render_ms = 0.
        self.last_ref_observed_frame = -1     # reference specifically, not any part
        self.last_diag_t = 0.
        self.render_error = None
        self.server = viser.ViserServer(host='127.0.0.1', port=args.port,
                                        label='Teach an articulated object')
        self.server.scene.set_up_direction('-y')
        self.server.gui.add_markdown('## Teach an articulated object\nPoint2pose · live estimates')
        self.info = self.server.gui.add_markdown('Waiting for RGB-D input.')
        self.image = self.server.gui.add_image(np.zeros((240,320,3), np.uint8), label='RGB-D tracking')
        self.mode = self.server.gui.add_dropdown('Demonstration', ['Scan surfaces', 'Move the lid'])
        self.advice = self.server.gui.add_markdown('Turn slowly while keeping the lid angle steady.')
        self.root = self.server.gui.add_dropdown('Reference body', ['0'])
        choose = self.server.gui.add_button('Use selected body as reference')
        self.object_view = self.server.gui.add_checkbox('Follow reference frame', initial_value=True)
        self.color = self.server.gui.add_checkbox('Color by part', initial_value=True)
        self.preview = self.server.gui.add_checkbox('Preview a frozen model', initial_value=False)
        self.joint = self.server.gui.add_dropdown('Preview part', ['None'])
        self.slider = self.server.gui.add_slider('Observed joint range', min=0., max=1., step=.01,
                                                 initial_value=.5)
        recenter = self.server.gui.add_button('Recenter view on model')
        save = self.server.gui.add_button('Save model version')
        reset = self.server.gui.add_button('Reset preview & annotations')
        self.saved = self.server.gui.add_markdown('Cloud + kinematic URDF when fitted. Observed surfaces only.')
        self.legend = self.server.gui.add_markdown('Legend appears once geometry is tracked.')
        self.uncertainty = self.server.gui.add_markdown('Tracking confidence appears with parts.')
        from examples.multi_part.render_dock import RenderDock
        self.render_dock = RenderDock(self.server)
        with self.render_dock.controls:
            self.diag = self.server.gui.add_checkbox('Render diagnostics', initial_value=False)
            self.diag_fps = self.server.gui.add_slider('Diagnostic target FPS', min=0.5, max=5., step=0.5, initial_value=5.)
            self.diag_panel = self.server.gui.add_dropdown('Diagnostic panel',
                ['Rendered RGB (approx)', 'Depth residual', 'Observed RGB'])
            self.diag_status = self.server.gui.add_markdown('Enable diagnostics for Gaussian render status.')
            self.diag_image = self.server.gui.add_image(np.zeros((240,320,3), np.uint8),
                label='Gaussian render diagnostic (not the point-cloud preview)')
            self.server.gui.add_markdown(
                '_Rendered RGB is an approximate per-part z-buffer composite (not joint '
                'alpha compositing). Observed RGB is the raw camera image, or the annotated '
                'tracking overlay if the raw frame is unavailable._')
        self._obs_is_overlay = False
        self.diag_panel.on_update(lambda _: self._show_diag())
        self.diag.on_update(lambda _: self.commands.append(('diagnostics', None)))
        self.server.gui.add_markdown('Select the physical body after parts separate. Reference pose is tracked, not fixed. Preview does not move the real object.')
        choose.on_click(lambda _: self.commands.append(('reference', int(self.root.value))))
        save.on_click(lambda _: self.save())
        recenter.on_click(lambda _: self.reframe())
        reset.on_click(lambda _: self.reset())
        for control in (self.object_view, self.color, self.slider, self.joint, self.mode):
            control.on_update(lambda _: self.draw())
        @self.preview.on_update
        def preview(_):
            with self.lock:
                self.frozen = self.state if self.preview.value else None
                self.draw()
        @self.server.on_client_connect
        def connected(client):
            self.reframe(client)
        print(f'[author] http://127.0.0.1:{args.port}', flush=True)

    def reframe(self, client=None):
        """Aim the camera at the current cloud centroid; a pure view change.

        Falls back to a sensible default framing before any geometry arrives.
        """
        with self.lock:
            if self.center is not None:
                cam = frame_view(self.center, self.radius)
            else:
                cam = dict(position=(0., -.35, -.55), look_at=(0., 0., .65),
                           up=(0., -1., 0.))
        try:
            clients = [client] if client is not None else list(self.server.get_clients().values())
        except Exception:
            clients = [client] if client is not None else []
        for c in clients:
            c.camera.position = cam['position']
            c.camera.look_at = cam['look_at']
            c.camera.up_direction = cam['up']

    def reset(self):
        """Clear the frozen preview and joint selection; mark a reset boundary.

        Scoped to the *view/annotation* layer: it drops any stale frozen model so
        it cannot leak into the live view, and records a 'reset' event rather than
        erasing provenance. It does not alter the tracked model, which the
        live/replay loop still owns.
        """
        with self.lock:
            frame = self.state.frame if self.state is not None else -1
            self.events.append(dict(frame=frame, action='reset', source=self.source))
            self.frozen = None
            self.preview.value = False
            if 'None' in self.joint.options:
                self.joint.value = 'None'          # only valid when no joint is fitted
            self.saved.content = 'Preview and annotations reset. Tracked model unchanged.'
            self.draw()

    def process_commands(self, stream):
        while self.commands:
            kind, pid = self.commands.popleft()
            if kind == 'diagnostics':
                self.last_diag_t = 0.
                self.last_update = 0.
                if not self.diag.value:
                    self.diag_status.content = 'Diagnostics paused. Enable Render diagnostics to render.'
                    self.render_dock.status('Diagnostics paused — previous images are stale.')
                continue
            try:
                stream.select_reference(pid)
                self.events.append(dict(frame=stream.n - 1, action=kind, part_id=pid,
                                        source=self.source))
            except ValueError as exc:
                self.saved.content = str(exc)

    def publish(self, stream, rgb, step_ms=0., tracking_valid=True, depth=None, raw_rgb=None):
        # `rgb` is the annotated tracking overlay (shown in the RGB-D panel).
        # `raw_rgb` is the unannotated camera image for the Observed RGB panel;
        # if None, that panel falls back to the overlay and is labelled as such.
        self.process_commands(stream)
        if time.monotonic() - self.last_update < .2:
            return
        self.last_update = time.monotonic()
        # Actual Gaussian render happens HERE, in the tracking thread (GPU-safe),
        # default-off and rate-limited -- never inside a GUI callback.
        if not self.diag.value:
            self.diag_status.content = 'Enable Render diagnostics to render.'
        if self.diag.value and time.monotonic() - self.last_diag_t >= 1. / self.diag_fps.value:
            self.last_diag_t = time.monotonic()
            self._render_diag(stream, depth, raw_rgb, rgb)
        with self.lock:
            self.state = snapshot(stream)
            if not tracking_valid:
                for p in self.state.parts:
                    p.observed = False
            if self.state.parts[self.state.root].observed:   # REFERENCE specifically
                self.last_ref_observed_frame = self.state.frame
            self.image.image = cv2.resize(rgb, (320,240))
            options = [str(p.part_id) for p in self.state.parts]
            if tuple(self.root.options) != tuple(options):
                self.root.options = options
            root = self.state.parts[self.state.root]
            counts = geometry_counts(self.state.labels, len(self.state.parts))
            self.info.content = (f'Frame **{self.state.frame}** · **{len(options)} parts** · '
                                 f'**{counts["assigned"]:,} assigned** / '
                                 f'{counts["unassigned"]:,} unassigned (hidden) pts\n\n'
                                 f'Reference **{root.part_id}**: '
                                 f'{"tracked" if root.observed else "LOST — held pose"} · '
                                 f'tracker step {step_ms:.0f} ms')
            self.uncertainty.content = uncertainty_markdown(self.state)
            model = getattr(stream, 'model', None)
            self.diag_status.content = render_status_markdown(
                model_present=model is not None and getattr(model, 'cloud', None) is not None,
                diag_enabled=self.diag.value,
                refiner_present=getattr(stream, 'refiner', None) is not None,
                counts=counts, last_render_frame=self.last_render_frame,
                last_render_ms=self.last_render_ms, current_frame=self.state.frame,
                frames_since_ref_observed=max(0, self.state.frame - self.last_ref_observed_frame),
                coverage=self.diag_coverage, agreement=self.diag_agreement,
                render_error=self.render_error)
            self._show_diag()
            self.draw()

    def _show_diag(self):
        """Swap which stored diagnostic image is shown. Pure display, no GPU."""
        img = {'Rendered RGB (approx)': self.diag_rgb, 'Depth residual': self.diag_resid,
               'Observed RGB': self.diag_obs}.get(self.diag_panel.value)
        if img is not None:
            self.diag_image.image = img

    def _render_diag(self, stream, depth, raw_rgb, overlay_rgb, tol=0.02):
        """Rasterise the actual Gaussians per part under its own pose, then
        z-buffer composite. This is an APPROXIMATE composite (nearest-depth, not
        true joint alpha compositing across parts); exact only for a single part.

        Tracking-thread only, under no_grad. Never fakes a result: on failure it
        records the error (surfaced in the UI) and keeps the previous images.
        Observed RGB uses the RAW camera image; if absent, the annotated overlay
        is shown and labelled as such."""
        model = getattr(stream, 'model', None)
        if model is None or getattr(model, 'cloud', None) is None:
            return
        try:
            import torch
            t0 = time.perf_counter()
            H, W = (depth.shape[:2] if depth is not None else
                    (raw_rgb if raw_rgb is not None else overlay_rgb).shape[:2])
            from examples.multi_part.render_views import render_part_views
            packet, reused = render_part_views(model, stream.parts, stream.K, H, W, stream.n - 1)
            H, W = packet['height'], packet['width']
            if depth is not None:
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            rgb_acc = np.zeros((H, W, 3), np.float32)
            z_acc = np.full((H, W), np.inf, np.float32)
            cover = np.zeros((H, W), bool)
            part_images = []
            observed = {p.part_id: p.observed for p in stream.parts}
            for view in packet['views']:
                rgb, dep, al = view['rgb'], view['depth'], view['alpha']
                good = (al > 0.3) & np.isfinite(dep) & (dep > 0)
                part_rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                part_images.append((f'Part {view["part_id"]} — {"observed" if observed[view["part_id"]] else "held pose"}', part_rgb))
                win = good & (dep < z_acc)
                rgb_acc[win] = rgb[win]; z_acc[win] = dep[win]; cover |= win
            self.diag_rgb = cv2.resize(np.clip(rgb_acc * 255, 0, 255).astype(np.uint8), (320, 240))
            obs = raw_rgb if raw_rgb is not None else overlay_rgb
            self.diag_obs = cv2.resize(obs, (320, 240)) if obs is not None else None
            self._obs_is_overlay = raw_rgb is None
            # Coverage (how much of the OBJECT the render draws) and depth
            # agreement (of covered object pixels, how many match observed depth)
            # are SEPARATE. The object is mask & finite depth, never background.
            self.diag_coverage = self.diag_agreement = self.diag_resid = None
            mask = getattr(stream, 'mask', None)
            if mask is not None:
                mask = cv2.resize(np.asarray(mask), (W, H), interpolation=cv2.INTER_NEAREST)
            if depth is not None and mask is not None and np.asarray(mask).shape == depth.shape:
                obj = (depth > 0) & np.isfinite(depth)
                if mask is not None and np.asarray(mask).shape[:2] == obj.shape:
                    obj &= np.asarray(mask) > 0
                nobj = int(obj.sum())
                if nobj:
                    self.diag_coverage = float((cover & obj).sum() / nobj)
                    valid = cover & obj
                    nv = int(valid.sum())
                    if nv:
                        res = np.zeros_like(depth, dtype=np.float32)
                        res[valid] = np.abs(z_acc[valid] - depth[valid])
                        self.diag_agreement = float((res[valid] < tol).sum() / nv)
                        heat = cv2.applyColorMap(
                            np.clip(res / (tol * 1.5) * 255, 0, 255).astype(np.uint8),
                            cv2.COLORMAP_MAGMA)
                        heat[~valid] = 0
                        self.diag_resid = cv2.resize(cv2.cvtColor(heat, cv2.COLOR_BGR2RGB), (320, 240))
            self.last_render_ms = 1000 * (time.perf_counter() - t0)
            self.last_render_frame = int(stream.n - 1)
            self.render_error = None
            if hasattr(self, 'render_dock'):
                self.render_dock.publish(self.last_render_frame, [('Overall (approx)', self.diag_rgb), ('Observed RGB', self.diag_obs), ('Depth residual', self.diag_resid)] + part_images, f'{W}×{H}; {"cached" if reused else "rasterized"}; {self.last_render_ms:.0f} ms; target {self.diag_fps.value:g} FPS')
        except Exception as exc:            # never fake a render; surface in UI + log
            self.render_error = f'{type(exc).__name__}: {exc}'
            print(f'[author] render diagnostic failed: {exc}', flush=True)
            if hasattr(self, 'render_dock'):
                self.render_dock.status(f'Render error: {exc}; previous images are stale.')

    def draw(self):
        with self.lock:
            s = self.frozen if self.preview.value else self.state
            if s is None:
                return
            candidates = {str(p.part_id): j for j,p in enumerate(s.parts)
                          if p.joint is not None and p.joint.kind in ('revolute','prismatic')}
            opts = list(candidates) or ['None']
            if tuple(self.joint.options) != tuple(opts):
                self.joint.options = opts
            selected = candidates.get(self.joint.value) if self.preview.value else None
            poses = displayed_poses(s, self.object_view.value, selected, self.slider.value)
            for node in self.nodes:
                node.remove()
            self.nodes = []
            shown, root_pts = [], None
            for j,p in enumerate(s.parts):
                ids = np.flatnonzero(s.labels == j)[::max(1, int(np.sum(s.labels == j) / 12000))]
                pts = s.points[ids] @ poses[j][:3,:3].T + poses[j][:3,3]
                colors = (np.tile(PALETTE[p.part_id % len(PALETTE)], (len(ids),1)) if self.color.value
                          else np.clip(s.colors[ids] * 255,0,255)).astype(np.uint8)
                self.nodes.append(self.server.scene.add_point_cloud(f'/parts/{p.part_id}',
                    points=pts.astype(np.float32), colors=colors, point_size=.003))
                if len(pts):
                    shown.append(pts)
                    if j == s.root:
                        root_pts = pts
            # Assigned-vs-unassigned counts + legend. Unassigned geometry
            # (label -1) is intentionally NOT drawn -- it has no part pose and the
            # viewer never invents one -- but its count is surfaced so the user
            # knows it exists. See doc: the viewer hides labels=-1.
            counts = geometry_counts(s.labels, len(s.parts))
            self.counts = counts
            self.legend.content = legend_markdown([p.part_id for p in s.parts], counts)
            # Framing/gizmo: centre on the geometry, not the distant anchor origin.
            if shown:
                self.center, self.radius = cloud_center_radius(np.concatenate(shown))
            T = poses[s.root]
            # Sit the reference gizmo on the reference body rather than at the
            # anchor origin, which is the initial camera pose ~0.5 m away.
            anchor = root_pts.mean(0) if root_pts is not None else T[:3,3]
            self.nodes.append(self.server.scene.add_frame('/reference', position=anchor,
                wxyz=Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]],
                axes_length=float(np.clip(0.6*self.radius, 0.03, 0.12)), axes_radius=.002))
            if not self.framed and shown:
                self.framed = True
                self.reframe()
            root = s.parts[s.root]
            if not root.observed:
                advice = 'Reference lost. Return toward the previous view before trusting new geometry.'
            elif self.preview.value:
                advice = 'Frozen model prediction. Scrub the observed range; unseen surfaces remain missing.'
            elif self.mode.value == 'Scan surfaces':
                advice = 'Turn the whole object slowly; keep the lid angle steady and views overlapping.'
            elif not candidates:
                advice = 'Move the lid while keeping the body visible. A hinge has not been fitted yet.'
            else:
                advice = 'Move the lid back and forth. Compare its tracked motion with the fitted model.'
            self.advice.content = advice

    def save(self):
        with self.lock:
            s = self.frozen if self.preview.value else self.state
            if s is None:
                return
            directory = Path(self.args.output) / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
            result = save_snapshot(s, directory, source=self.source)
            (directory / 'events.json').write_text(json.dumps(self.events, indent=2))
            self.saved.content = f'Saved `{directory}`. ' + ('URDF included.' if result['urdf'] else 'Cloud saved; joint tree is not fitted.')
            print(f'[author] saved {directory}', flush=True)


def add_args(ap):
    ap.add_argument('--port', type=int, default=8088)
    ap.add_argument('--output', default='results/author_demo_v1')


def make_tracker(K, cfg, checkpoint):
    from point2pose.modules.tracker.tapir_tracker import TapirTracker
    from point2pose.modules.register.svd_cluster_ransac_register import SVDClusterRANSACRegister
    tracker = TapirTracker(dict(checkpoint_path=checkpoint, resize_height=cfg.tapir_res,
        resize_width=cfg.tapir_res, num_pips_iter=cfg.num_pips_iter, visible_threshold=.5, device='cuda'))
    reg = SVDClusterRANSACRegister(dict(ransac_iters=cfg.ransac_iters, sample_size=4,
        inlier_thres=cfg.inlier_thres, min_inliers=cfg.min_inliers, max_clusters=4, use_uncertainty=False))
    cfg.reroot = True
    return ReferenceTracker(K, cfg, tracker, reg)


def replay():
    from examples.multi_part.recording import Recording
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq-dir', required=True)
    ap.add_argument('--config', default='reprojection_split.yaml')
    ap.add_argument('--checkpoint', default='checkpoints/tapir/causal_bootstapir_checkpoint.pt')
    ap.add_argument('--set', action='append', default=[])
    ap.add_argument('--exit-after-replay', action='store_true')
    add_args(ap)
    args = ap.parse_args()
    r = Recording(args.seq_dir)
    cfg = apply_config(NaiveConfig(), args.config)
    apply_overrides(cfg,args.set)
    stream = make_tracker(r.K,cfg,args.checkpoint)
    view = AuthorView(args)
    view.source = 'replay'
    trace = []
    for i in range(len(r)):
        rgb, depth, mask = r.get_color(i), r.get_depth(i), r.get_mask(i)
        if mask is None or not mask.any():
            raise ValueError(f'Replay requires an object mask on frame {i}; no GT labels are loaded')
        view.process_commands(stream) if i else None
        t = time.perf_counter()
        if i == 0:
            stream.start(rgb,depth,mask)
            stream.root = 0
        else:
            stream.step(rgb,depth,mask)
        view.publish(stream,cv2.cvtColor(stream.render(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR), style='clean'),cv2.COLOR_BGR2RGB),
                     1000*(time.perf_counter()-t), depth=depth, raw_rgb=rgb)
        root = stream.parts[stream.root]
        trace.append(dict(frame=i, part_id=root.part_id, observed=bool(root.observed),
                          pose=root.pose.tolist(), parts=len(stream.parts)))
    view.last_update = 0
    view.publish(stream,rgb, depth=depth, raw_rgb=rgb)
    view.save()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / 'reference_trace.json').write_text(json.dumps(trace,allow_nan=False))
    print('[author] replay complete; browser remains interactive',flush=True)
    if not args.exit_after_replay:
        while True:
            if view.commands or (view.diag.value and time.monotonic() - view.last_diag_t >= 1. / view.diag_fps.value):
                view.last_update = 0
                view.publish(stream, rgb, depth=depth, raw_rgb=rgb)
            time.sleep(.1)
    view.render_dock.close()
    view.server.stop()


def live():
    from examples.multi_part.live_acquire import LiveAcquire, main
    class LiveAuthor(LiveAcquire):
        def __init__(self,args):
            self.author = AuthorView(args)
            super().__init__(args)

        def _make_stream(self,cfg,tracker,reg):
            cfg.reroot = True
            stream = ReferenceTracker(self.K,cfg,tracker,reg)
            directory = Path(self.args.output) / ('capture-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
            for sub in ('rgb', 'depth', 'masks'):
                (directory / sub).mkdir(parents=True, exist_ok=False)
            np.savetxt(directory / 'cam_K.txt', self.K)
            count = 0
            def record(rgb, depth, mask):
                nonlocal count
                stream.author_input = (rgb.copy(), depth.copy())
                filename = f'{count:06d}.png'
                for sub, data in [('rgb', cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)),
                                  ('depth', np.clip(np.nan_to_num(depth)*1000,0,65535).astype(np.uint16)),
                                  ('masks', (mask > 0).astype(np.uint8))]:
                    if not cv2.imwrite(str(directory / sub / filename), data):
                        raise IOError(f'Could not save {sub}/{filename}')
                count += 1
            stream.recorder = record
            print(f'[author] recording processed RGB-D frames to {directory}',flush=True)
            return stream

        def _panel(self,vis,conf,kind,i):
            if self.stream is not None:
                if not hasattr(self.stream,'root'):
                    self.stream.root = 0
                self.author.publish(self.stream,cv2.cvtColor(vis,cv2.COLOR_BGR2RGB),
                                    self.times[-1] if self.times else 0.,
                                    tracking_valid=not self.acq.tracking_lost,
                                    raw_rgb=getattr(self.stream, "author_input", (None, None))[0],
                                    depth=getattr(self.stream, "author_input", (None, None))[1])
            return super()._panel(vis,conf,kind,i)

        def run(self):
            try:
                super().run()
            finally:
                self.author.save()
                self.author.render_dock.close()
                self.author.server.stop()
    main(LiveAuthor,add_args)


if __name__ == '__main__':
    import sys
    replay() if '--seq-dir' in sys.argv else live()

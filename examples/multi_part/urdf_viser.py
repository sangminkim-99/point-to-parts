"""Interactive viewer for author-demo object.urdf + model.npz exports.

python -m examples.multi_part.urdf_viser <export-directory> --port 8091
"""
import argparse
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from examples.multi_part.urdf_view import read_urdf, link_poses


def covariance(scales, quats):
    """Stored quaternions are wxyz; scales are standard deviations in metres."""
    R = Rotation.from_quat(np.asarray(quats)[:, [1, 2, 3, 0]]).as_matrix()
    return (R * np.asarray(scales)[:, None, :] ** 2) @ R.transpose(0, 2, 1)


def load_export(path, model_path=None):
    path = Path(path)
    urdf = path / 'object.urdf' if path.is_dir() else path
    model_path = Path(model_path) if model_path else urdf.parent / 'model.npz'
    links, joints, root = read_urdf(urdf)
    with np.load(model_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    for k in ('means', 'colors', 'labels', 'poses', 'part_ids'):
        if k not in data:
            raise ValueError(f'{model_path}: missing {k}')
    for j in range(len(data['part_ids'])):
        if f'part{j}' not in links:
            raise ValueError(f'URDF has no part{j}; model and URDF must be matching exports')
    if root not in [f'part{j}' for j in range(len(data['part_ids']))]:
        raise ValueError('Expected an author-demo part root')
    if any(j['type'] not in ('fixed', 'prismatic', 'revolute') for j in joints):
        raise ValueError('This viewer supports exported fixed, revolute and prismatic joints')
    return data, joints, root


class ExportViewer:
    def __init__(self, server, data, joints, root):
        self.server, self.data, self.joints, self.root = server, data, joints, root
        self.base = data['poses'][int(root.removeprefix('part'))]
        self.q = np.array([np.clip(0., j['lower'], j['upper']) for j in joints])
        self.frames, self.points, self.splats, self.sliders = {}, [], [], []
        have_gaussians = all(k in data for k in ('scales', 'quats', 'opacities'))
        cov = covariance(data['scales'], data['quats']) if have_gaussians else None
        panel = server.gui.add_panel()
        panel.dock_left()
        panel.set_width(340)
        with panel.add_tab('Model & joints'):
            server.gui.add_markdown('## Exported articulated model\nMove joints within their observed ranges. Orbit and zoom in the 3D view.')
            self.mode = server.gui.add_dropdown('Geometry', options=('Gaussians', 'Points') if have_gaussians else ('Points',), initial_value='Gaussians' if have_gaussians else 'Points')
            self.pose_mode = server.gui.add_dropdown('Pose source', options=('Saved tracked poses', 'URDF joint poses'), initial_value='Saved tracked poses')
            server.gui.add_markdown('Saved poses reproduce the capture. URDF poses follow the fitted joints and may differ because of fit error. Joint sliders start at zero clipped to observed limits, not the saved joint angle.')
            self.axes = server.gui.add_checkbox('Link frames', initial_value=False)
            hidden = int(np.sum((data['labels'] < 0) | (data['labels'] >= len(data['part_ids']))))
            server.gui.add_markdown(f"{len(data['part_ids'])} parts · {hidden:,} unassigned points hidden.\n\n" +
                ('Native browser Gaussian rendering; appearance can differ from gsplat.' if have_gaussians else 'Legacy point-only export. Save again in the updated author demo to retain Gaussian parameters.'))
            for k, j in enumerate(joints):
                if j['type'] == 'fixed' or j['upper'] <= j['lower']:
                    continue
                angular = j['type'] == 'revolute'
                factor = 180 / np.pi if angular else 1000.
                lo, hi = j['lower'] * factor, j['upper'] * factor
                h = server.gui.add_slider(f"{j['name']} ({'deg' if angular else 'mm'})", min=lo, max=hi, step=max((hi-lo)/500, 1e-6), initial_value=self.q[k]*factor)
                def change(event, k=k, factor=factor):
                    self.q[k] = event.target.value / factor
                    self.update()
                h.on_update(change)
                self.sliders.append(h)
        for j, pid in enumerate(data['part_ids']):
            name = f'/model/part{j}'
            self.frames[j] = server.scene.add_frame(name, show_axes=False, axes_length=.05, axes_radius=.001)
            sel = data['labels'] == j
            if not sel.any():
                continue
            self.points.append(server.scene.add_point_cloud(name+'/points', points=data['means'][sel], colors=data['colors'][sel], point_size=.003, visible=not have_gaussians))
            if have_gaussians:
                self.splats.append(server.scene.add_gaussian_splats(name+'/gaussians', centers=data['means'][sel], covariances=cov[sel], rgbs=data['colors'][sel], opacities=data['opacities'][sel].reshape(-1,1)))
        self.mode.on_update(lambda _: self.update())
        self.axes.on_update(lambda _: self.update())
        self.pose_mode.on_update(lambda _: self.update())
        self.update()
        @server.on_client_connect
        def connected(client):
            poses = self.poses()
            points = [data['means'][data['labels']==j] @ poses[f'part{j}'][:3,:3].T + poses[f'part{j}'][:3,3] for j in self.frames if np.any(data['labels']==j)]
            if points:
                p = np.concatenate(points)
                center = np.median(p, axis=0)
                radius = max(.1, float(np.percentile(np.linalg.norm(p-center, axis=1), 95)))
                client.camera.up_direction = (0., -1., 0.)
                client.camera.look_at = center
                client.camera.position = center + np.array([.3,-.3,-2.8])*radius

    def poses(self):
        if hasattr(self, 'pose_mode') and self.pose_mode.value == 'Saved tracked poses':
            return {f'part{j}': T for j,T in enumerate(self.data['poses'])}
        return {name: self.base @ T for name, T in link_poses(self.joints, self.root, self.q).items()}

    def update(self):
        with self.server.atomic():
            for slider in self.sliders:
                slider.disabled = self.pose_mode.value == 'Saved tracked poses'
            for j, handle in self.frames.items():
                T = self.poses()[f'part{j}']
                handle.position = T[:3, 3]
                handle.wxyz = Rotation.from_matrix(T[:3,:3]).as_quat()[[3,0,1,2]]
                handle.show_axes = self.axes.value
            for h in self.points:
                h.visible = self.mode.value == 'Points'
            for h in self.splats:
                h.visible = self.mode.value == 'Gaussians'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('export', help='Export directory or object.urdf')
    ap.add_argument('--model', help='Matching model.npz (default: beside URDF)')
    ap.add_argument('--port', type=int, default=8091)
    args = ap.parse_args()
    data, joints, root = load_export(args.export, args.model)
    import viser
    server = viser.ViserServer(port=args.port)
    ExportViewer(server, data, joints, root)
    print(f'URDF + model viewer: http://localhost:{args.port}', flush=True)
    try:
        while True:
            time.sleep(.5)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()

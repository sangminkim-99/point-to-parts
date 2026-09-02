"""Interactive 3D inspector for an RBO sequence.

Serves a viser scene with a frame slider so the sequence can be scrubbed:
back-projected RGB-D point cloud, SAM2 or GT part masks used as colouring,
point-track trajectories, and the GT part meshes for comparison.

The point of it is to see, rather than infer, three things that the numbers
only summarise: how large the depth error actually is (GT mesh against the
observed cloud), when a part visibly separates from its parent, and whether the
tracks sit on the surface they are supposed to.
"""

import argparse
import os
import sys
import time

import numpy as np
import viser

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.io.sources.dataset.sapien_reader import open_sequence

PART_COLORS = np.array([
    [245, 135, 66], [60, 130, 214], [100, 190, 90],
    [220, 100, 170], [230, 200, 80], [250, 120, 120],
], dtype=np.uint8)


def backproject(depth, rgb, K, stride=2, zmin=0.1, zmax=3.0):
    H, W = depth.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    z = depth[::stride, ::stride]
    ok = (z > zmin) & (z < zmax) & np.isfinite(z)
    us, vs, z = us[ok], vs[ok], z[ok]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=1).astype(np.float32)
    cols = rgb[::stride, ::stride][ok].astype(np.uint8)
    return pts, cols, (us, vs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--masks", default=None,
                    help="npz from run_sam2_propagate.py --save-masks")
    ap.add_argument("--tracks", default=None,
                    help="npz from demo_part_discovery.py --save-tracks")
    ap.add_argument("--stride", type=int, default=3, help="frame stride")
    ap.add_argument("--pixel-stride", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=150)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    reader = open_sequence(args.seq_dir)
    parts = reader.get_object_names()
    frames = list(range(0, len(reader), args.stride))[: args.max_frames]

    mask_data = None
    if args.masks and os.path.exists(args.masks):
        d = np.load(args.masks, allow_pickle=True)
        mask_data = {int(f): d["masks"][i] for i, f in enumerate(d["frames"])
                     if i < len(d["masks"])}
        mask_names = [str(x) for x in d["names"]]
        print(f"[viser] loaded SAM2 masks for {len(mask_data)} frames: {mask_names}")

    tracks = None
    if args.tracks and os.path.exists(args.tracks):
        t = np.load(args.tracks, allow_pickle=True)
        tracks = {"xyz": t["xyz"], "frames": t["frames"],
                  "valid": t["valid"], "visible": t["visible"],
                  "gt": t["gt_label"]}
        print(f"[viser] loaded {tracks['xyz'].shape[1]} tracks over "
              f"{tracks['xyz'].shape[0]} frames")

    print("[viser] pre-loading frames ...")
    cache = {}
    for i in frames:
        depth, rgb = reader.get_depth(i), reader.get_color(i)
        pts, cols, (us, vs) = backproject(depth, rgb, reader.K, args.pixel_stride)
        part_cols = None
        if mask_data is not None and i in mask_data:
            m = mask_data[i]
            lab = np.full(len(pts), -1, np.int16)
            for k in range(len(m)):
                sel = m[k][vs, us] > 0
                lab[sel] = k
            part_cols = np.where(
                (lab >= 0)[:, None],
                PART_COLORS[np.clip(lab, 0, len(PART_COLORS) - 1) % len(PART_COLORS)],
                np.array([90, 90, 90], np.uint8),
            ).astype(np.uint8)
        cache[i] = (pts, cols, part_cols)
    print(f"[viser] cached {len(cache)} frames")

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("-y")          # camera-frame convention

    with server.gui.add_folder("Sequence"):
        gui_frame = server.gui.add_slider(
            "frame", min=0, max=len(frames) - 1, step=1,
            initial_value=min(25, len(frames) - 1))
        gui_play = server.gui.add_checkbox("play", False)
        gui_info = server.gui.add_text("state", initial_value="", disabled=True)
    with server.gui.add_folder("Show"):
        # default to the masks when the user bothered to pass them
        gui_color = server.gui.add_dropdown(
            "point cloud colour", ("rgb", "part mask"),
            initial_value="part mask" if mask_data is not None else "rgb")
        gui_size = server.gui.add_slider("point size", 0.001, 0.02, 0.001, 0.004)
        gui_mesh = server.gui.add_checkbox("GT part meshes", True)
        gui_tracks = server.gui.add_checkbox("point tracks", tracks is not None)
        gui_trail = server.gui.add_slider("track trail", 0, 60, 1, 20)
        gui_lw = server.gui.add_slider("track line width", 0.5, 8.0, 0.5, 3.0)

    meshes = reader._part_meshes() if len(parts) else {}

    def render(idx):
        i = frames[idx]
        pts, cols, part_cols = cache[i]
        use = part_cols if (gui_color.value == "part mask" and part_cols is not None) else cols
        server.scene.add_point_cloud("/cloud", points=pts, colors=use,
                                     point_size=gui_size.value)

        for k, p in enumerate(parts):
            name = f"/gt/{p}"
            if not gui_mesh.value or p not in meshes:
                server.scene.add_frame(name, show_axes=False)
                continue
            T = reader.get_gt_pose(i, p)
            if T is None:
                continue
            m = meshes[p].copy()
            m.apply_transform(T)
            c = PART_COLORS[k % len(PART_COLORS)]
            server.scene.add_mesh_simple(
                name, vertices=np.asarray(m.vertices, np.float32),
                faces=np.asarray(m.faces, np.uint32),
                color=(int(c[0]), int(c[1]), int(c[2])),
                opacity=0.35, wireframe=False)

        n_seg = 0
        if tracks is not None and gui_tracks.value:
            fr = list(tracks["frames"])
            if i in fr:
                t = fr.index(i)
                lo = max(0, t - gui_trail.value)
                seg, colr = [], []
                xyz = tracks["xyz"]
                for j in range(xyz.shape[1]):
                    v = tracks["valid"][lo:t + 1, j] & tracks["visible"][lo:t + 1, j]
                    if v.sum() < 2:
                        continue
                    p = xyz[lo:t + 1, j][v]
                    seg.append(np.stack([p[:-1], p[1:]], axis=1))
                    g = int(tracks["gt"][j])
                    c = PART_COLORS[g % len(PART_COLORS)] if g >= 0 else np.array([120, 120, 120])
                    # viser wants a colour per endpoint: (N, 2, 3), not (N, 3)
                    colr.append(np.tile(c, (len(p) - 1, 2, 1)))
                n_seg = len(seg)
                if seg:
                    pts_seg = np.concatenate(seg).astype(np.float32)
                    col_seg = np.concatenate(colr).astype(np.uint8)
                    server.scene.add_line_segments(
                        "/tracks", points=pts_seg, colors=col_seg,
                        line_width=gui_lw.value)
                    # current head of each track, so the trail is easy to find
                    heads = pts_seg[:, 1][
                        np.unique(np.cumsum([len(x) for x in seg]) - 1)
                    ]
                    server.scene.add_point_cloud(
                        "/tracks_head", points=heads,
                        colors=col_seg[:, 1][
                            np.unique(np.cumsum([len(x) for x in seg]) - 1)
                        ],
                        point_size=gui_size.value * 1.8)
                else:
                    server.scene.add_point_cloud(
                        "/tracks_head", points=np.zeros((0, 3), np.float32),
                        colors=np.zeros((0, 3), np.uint8), point_size=0.001)

        js = reader.get_joint_states(i)
        jt = {j["name"]: j["type"] for j in reader.joints}
        txt = "  ".join(
            f"{k}={np.degrees(v):.0f}deg" if jt.get(k) == "revolute" else f"{k}={v*1000:.0f}mm"
            for k, v in js.items())
        mask_txt = ""
        if mask_data is not None:
            if i in mask_data:
                cov = [int(m.sum()) for m in mask_data[i]]
                mask_txt = f"   masks px: {cov}"
            else:
                mask_txt = "   masks: none for this frame"
        trail_txt = ""
        if tracks is not None:
            trail_txt = f"   tracks: {n_seg} trails"
            if gui_tracks.value and n_seg == 0:
                trail_txt += " (need >=2 past frames - move the slider forward)"
        gui_info.value = f"frame {i}   {txt}{mask_txt}{trail_txt}"

    for g in (gui_frame, gui_color, gui_size, gui_mesh, gui_tracks, gui_trail, gui_lw):
        g.on_update(lambda _: render(gui_frame.value))

    render(gui_frame.value)
    print(f"\n[viser] serving on http://localhost:{args.port}  (Ctrl-C to stop)")
    while True:
        if gui_play.value:
            gui_frame.value = (gui_frame.value + 1) % len(frames)
            render(gui_frame.value)
            time.sleep(0.08)
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()

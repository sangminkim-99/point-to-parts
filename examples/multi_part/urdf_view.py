"""Look at an exported URDF: drive every joint through the range we observed.

A URDF is only as good as what it does when you move it, and the failures that
matter are the ones you see rather than measure -- a drawer that slides out
sideways, a lid that hinges about an axis floating in the air, a collision mesh
larger than the body it belongs to. So this drives each joint in turn over
exactly the limits the export claimed and either shows it or writes a video.

    python -m examples.multi_part.urdf_view results/urdf/ikeasmall02.urdf
    python -m examples.multi_part.urdf_view <path>.urdf --out anim.mp4

Reads the URDF with the standard library; the only real dependency is Open3D,
which the export already needs for the meshes.
"""
import argparse
import os
import xml.etree.ElementTree as ET

import numpy as np

def cv2_bgr(rgb):
    import cv2
    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


PAL = [(0.90, 0.42, 0.24), (0.24, 0.52, 0.85), (0.36, 0.70, 0.42),
       (0.85, 0.72, 0.25), (0.66, 0.40, 0.78), (0.30, 0.72, 0.72)]


def _vec(node, attr, default):
    if node is None or node.get(attr) is None:
        return np.asarray(default, float)
    return np.asarray([float(x) for x in node.get(attr).split()], float)


def read_urdf(path):
    """Links with their mesh paths, and joints with axis, origin and limits."""
    root = ET.parse(path).getroot()
    base = os.path.dirname(os.path.abspath(path))
    links = {}
    for ln in root.findall("link"):
        mesh = ln.find("./visual/geometry/mesh")
        links[ln.get("name")] = (None if mesh is None else
                                 os.path.join(base, mesh.get("filename")))
    joints = []
    for jn in root.findall("joint"):
        lim = jn.find("limit")
        joints.append({
            "name": jn.get("name"),
            "type": jn.get("type"),
            "parent": jn.find("parent").get("link"),
            "child": jn.find("child").get("link"),
            "origin": _vec(jn.find("origin"), "xyz", [0, 0, 0]),
            "axis": _vec(jn.find("axis"), "xyz", [0, 0, 1]),
            "lower": 0.0 if lim is None else float(lim.get("lower", 0.0)),
            "upper": 0.0 if lim is None else float(lim.get("upper", 0.0)),
        })
    child_of = {j["child"] for j in joints}
    roots = [n for n in links if n not in child_of]
    return links, joints, (roots[0] if roots else next(iter(links)))


def joint_pose(j, q):
    """The child's transform relative to its parent at joint value q."""
    T = np.eye(4)
    a = j["axis"] / max(np.linalg.norm(j["axis"]), 1e-9)
    if j["type"] == "prismatic":
        T[:3, 3] = a * q
    elif j["type"] == "revolute":
        # Rodrigues about the axis, then carried around the joint's origin so
        # the link swings about the hinge instead of about the world centre
        K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * (K @ K)
        T[:3, :3] = R
        T[:3, 3] = j["origin"] - R @ j["origin"]
    return T


def link_poses(joints, root, q):
    """World transform per link, walking down from the root."""
    pose = {root: np.eye(4)}
    todo, guard = list(joints), 0
    while todo and guard < 100:
        guard += 1
        left = []
        for k, j in enumerate(todo):
            if j["parent"] in pose:
                pose[j["child"]] = pose[j["parent"]] @ joint_pose(j, q[k])
            else:
                left.append(j)
        if len(left) == len(todo):
            break                       # a joint whose parent never appears
        todo = left
    return pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("--out", default=None,
                    help="write an mp4 instead of opening a window")
    ap.add_argument("--frames", type=int, default=48,
                    help="frames per joint sweep")
    ap.add_argument("--size", type=int, nargs=2, default=[900, 700])
    ap.add_argument("--seq-dir", default=None,
                    help="the sequence this URDF came from. The exported "
                         "geometry is already in camera coordinates, so with "
                         "the sequence's own intrinsics the animation is drawn "
                         "from the viewpoint that recorded it, over the frame "
                         "it was recorded in -- which is the only view in "
                         "which it can be compared with the demo.")
    ap.add_argument("--bg-frame", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    args = ap.parse_args()

    import open3d as o3d
    links, joints, root = read_urdf(args.urdf)
    print(f"[urdf-view] {os.path.basename(args.urdf)}: "
          f"{len(links)} links, {len(joints)} joints, root {root}")
    for j in joints:
        span = j["upper"] - j["lower"]
        unit = ("deg", np.degrees(span)) if j["type"] == "revolute" \
            else ("mm", span * 1000)
        print(f"[urdf-view]   {j['name']}: {j['type']} "
              f"{j['parent']} -> {j['child']}, "
              f"observed range {unit[1]:.1f} {unit[0]}, "
              f"axis {np.round(j['axis'], 3)}")

    geoms, order = {}, []
    for i, (name, mesh_path) in enumerate(links.items()):
        if mesh_path is None or not os.path.exists(mesh_path):
            print(f"[urdf-view]   {name}: no mesh at {mesh_path}")
            continue
        m = o3d.io.read_triangle_mesh(mesh_path)
        m.compute_vertex_normals()
        m.paint_uniform_color(PAL[i % len(PAL)])
        geoms[name] = m
        order.append(name)
    if not geoms:
        print("[urdf-view] nothing to draw")
        return

    # centre the whole thing on its own root: the export writes geometry in the
    # anchor camera's frame, which puts the object a metre in front of nothing
    allpts = np.vstack([np.asarray(m.vertices) for m in geoms.values()])
    centre = allpts.mean(0)
    radius = float(np.linalg.norm(allpts - centre, axis=1).max())
    rest = {n: np.asarray(m.vertices).copy() for n, m in geoms.items()}

    # one sweep per joint, the others held at rest, then all together
    plans = []
    for k in range(len(joints)):
        plans.append(("%s alone" % joints[k]["name"], k))
    if len(joints) > 1:
        plans.append(("all together", None))

    def q_at(t, only):
        q = np.zeros(len(joints))
        for k, j in enumerate(joints):
            if only is not None and k != only:
                continue
            # out and back, so the video ends where it started
            u = 0.5 - 0.5 * np.cos(2 * np.pi * t)
            q[k] = j["lower"] + u * (j["upper"] - j["lower"])
        return q

    def apply(q):
        pose = link_poses(joints, root, q)
        for n, m in geoms.items():
            T = pose.get(n, np.eye(4))
            v = rest[n] @ T[:3, :3].T + T[:3, 3] - centre
            m.vertices = o3d.utility.Vector3dVector(v)
            m.compute_vertex_normals()

    W, H = args.size
    K, bg = None, None
    if args.seq_dir:
        try:
            from point2pose.io.sources.dataset.rbo_view import _noop  # noqa
        except Exception:
            pass
        try:
            from point2pose.io.sources.dataset.rbo_reader import RBOReader
            rr = RBOReader(args.seq_dir)
            K = np.asarray(rr.K, float)
            bg = cv2_bgr(rr.get_color(min(args.bg_frame, len(rr) - 1)))
        except Exception:
            from examples.multi_part.recording import Recording
            rr = Recording(args.seq_dir)
            K = np.asarray(rr.K, float)
            bg = cv2_bgr(rr.get_color(min(args.bg_frame, len(rr) - 1)))
        H, W = bg.shape[:2]

    if args.out:
        # Drawn here rather than through OpenGL: a headless GL context on this
        # machine returns a frame that is not what the viewer shows, and a
        # URDF check is worthless if you cannot trust the picture. Vertices are
        # projected and splatted back to front, which is enough to see whether
        # a drawer slides along its own body and where the axis runs.
        import cv2
        if K is not None:
            f, cx, cy = K[0, 0], K[0, 2], K[1, 2]
            eye = np.zeros(3)
            Rc = np.eye(3)               # the geometry is already in camera
            off = np.zeros(3)
        else:
            f, cx, cy = 1.6 * max(W, H), W / 2, H / 2
            eye = np.array([0.9, -0.6, -1.9]) * radius * 2.2
            fwd = -eye / np.linalg.norm(eye)
            right = np.cross(fwd, [0, -1, 0]); right /= np.linalg.norm(right)
            Rc = np.stack([right, np.cross(right, fwd), fwd])
            off = centre

        def draw(q, label):
            img = (bg.copy() * 0.35).astype(np.uint8) if bg is not None \
                else np.full((H, W, 3), (26, 30, 34), np.uint8)
            pose = link_poses(joints, root, q)
            pts, col, dep = [], [], []
            for i, (n, m) in enumerate(geoms.items()):
                T = pose.get(n, np.eye(4))
                v = rest[n] @ T[:3, :3].T + T[:3, 3] - off
                c = Rc @ (v - eye).T
                z = c[2]
                ok = z > 1e-3
                pts.append(np.stack([f * c[0][ok] / z[ok] + cx,
                                     f * c[1][ok] / z[ok] + cy], 1))
                dep.append(z[ok]); col.append(np.full(int(ok.sum()), i))
            P = np.concatenate(pts); D = np.concatenate(dep)
            C = np.concatenate(col).astype(int)
            o = np.argsort(-D)
            P, D, C = P[o], D[o], C[o]
            keep = (P[:, 0] > -20) & (P[:, 0] < W + 20) & \
                   (P[:, 1] > -20) & (P[:, 1] < H + 20)
            P, D, C = P[keep], D[keep], C[keep]
            if D.size:
                lo, hi = D.min(), D.max()
                shade = 1.15 - 0.55 * (D - lo) / max(hi - lo, 1e-6)
                for (x, y), s_, ci in zip(P.astype(int), shade, C):
                    b, g, r = PAL[ci % len(PAL)][::-1]
                    cv2.circle(img, (x, y), 2,
                               (int(255 * min(b * s_, 1)),
                                int(255 * min(g * s_, 1)),
                                int(255 * min(r * s_, 1))), -1)
            for k, j in enumerate(joints):
                Tp = pose.get(j["parent"], np.eye(4))
                a = Tp[:3, :3] @ (j["axis"] / np.linalg.norm(j["axis"]))
                if j["type"] == "revolute":
                    b0 = Tp[:3, :3] @ j["origin"] + Tp[:3, 3] - off
                else:
                    ch = geoms.get(j["child"])
                    b0 = (np.asarray(ch.vertices).mean(0) if ch is not None
                          else Tp[:3, 3] - off)
                seg = np.stack([b0 - a * radius * 0.7, b0 + a * radius * 0.7])
                cc = Rc @ (seg - eye).T
                if (cc[2] <= 1e-3).any():
                    continue
                uv = np.stack([f * cc[0] / cc[2] + cx,
                               f * cc[1] / cc[2] + cy], 1).astype(int)
                cv2.line(img, tuple(uv[0]), tuple(uv[1]), (20, 20, 20), 5,
                         cv2.LINE_AA)
                cv2.line(img, tuple(uv[0]), tuple(uv[1]), (240, 240, 240), 2,
                         cv2.LINE_AA)
                t = f"{j['name']} {j['type']}  q={q[k]:+.3f}"
                for c_, w_ in (((20, 20, 20), 3), ((235, 235, 235), 1)):
                    cv2.putText(img, t, (uv[1][0] + 8, uv[1][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, c_, w_,
                                cv2.LINE_AA)
            cv2.rectangle(img, (0, H - 30), (W, H), (18, 20, 23), -1)
            cv2.putText(img, label, (12, H - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (225, 225, 225), 1, cv2.LINE_AA)
            return img

        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                 args.fps, (W, H))
        wrote = 0
        for label, only in plans:
            for fr in range(args.frames):
                writer.write(draw(q_at(fr / args.frames, only), label))
                wrote += 1
        writer.release()
        print(f"[urdf-view] wrote {args.out} ({wrote} frames, {len(plans)} "
              f"sweeps, object radius {radius * 100:.0f} cm"
              f"{', from the recording camera' if K is not None else ''})")
        return

    # interactive: space steps through the sweeps, the joints animate on a timer
    state = {"t": 0.0, "plan": 0}
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=W, height=H,
                      window_name=os.path.basename(args.urdf))
    apply(np.zeros(len(joints)))
    for m in geoms.values():
        vis.add_geometry(m)
    vis.reset_view_point(True)
    vis.get_render_option().background_color = np.array([0.10, 0.12, 0.14])

    def step(v):
        state["t"] = (state["t"] + 1.0 / args.frames) % 1.0
        apply(q_at(state["t"], plans[state["plan"]][1]))
        for m in geoms.values():
            v.update_geometry(m)
        return False

    def nxt(v):
        state["plan"] = (state["plan"] + 1) % len(plans)
        print(f"[urdf-view] {plans[state['plan']][0]}")
        return False

    vis.register_animation_callback(step)
    vis.register_key_callback(ord(" "), nxt)
    print("[urdf-view] space: next joint sweep, q: quit")
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()

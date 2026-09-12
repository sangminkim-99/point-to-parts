"""Thin-slab turnover control: fixed camera, whole-object flip through edge-on.

Why this exists: the user turns a thin book over in front of a fixed RealSense.
Near edge-on the visible mask collapses to a sliver a few millimetres thick,
depth support vanishes, and the honest behaviours under test are (a) the
tracker holds rather than fabricates the pose, and (b) it reports held, not
observed. No existing control passes through edge-on: `moving_laptop` yaws 20
degrees in plane and never loses the face.

Variants:
    rigid   one thin slab (book held closed), turned over ~180 deg about an
            in-plane axis. GT is ONE part: any split is false.
    hinge   body slab + thin cover joined by a revolute joint at the spine.
            The object is turned over first, then the cover opens (sequential
            by default, --overlap makes them simultaneous). GT is TWO parts
            and one revolute joint.

Outputs exactly the conventions of scripts/sim/render_partnet_sequence.py, so
SapienReader / prep_demo_input / eval_moving_root consume it unchanged:
    rgb/000000.png        uint8
    depth/000000.png      uint16 millimetres, 0 = invalid
    seg/000000.png        uint8 part index, 255 = background   (oracle labels)
    meta.json             intrinsics, parts, joints, render settings, edge-on stats
    poses.npz             T_cam_part (T,K,4,4), joint_states (T,J), cam_poses,
                          timestamps, mask_px (T,K)

GT poses and per-part seg indices are EVAL-ONLY: the tracker receives rgb,
depth and the binary union mask (seg != 255) only. The camera is fixed by
construction -- there is no orbit option -- because the moving object with a
fixed camera is the point of the control.
"""
import argparse
import json
import os
import numpy as np


def window_profile(n, amount, a, b):
    """0 -> amount with cosine ease inside the fractional window [a, b].

    Flat at 0 before `a`, flat at `amount` after `b`. The flat prefix matters
    for the same reason as the renderer's static_prefix: a correct method must
    refuse to split (or to claim motion) while nothing moves.
    """
    if not (0.0 <= a < b <= 1.0):
        raise ValueError("window must satisfy 0 <= a < b <= 1")
    if n < 2:
        raise ValueError("need at least 2 frames")
    t = np.arange(n) / (n - 1)
    s = np.clip((t - a) / (b - a), 0.0, 1.0)
    return amount * (0.5 - 0.5 * np.cos(np.pi * s))


def parse_window(text):
    a, b = (float(x) for x in text.split(","))
    return a, b


def _axis_vector(name):
    return {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}[name]


def _rot_about(axis, theta):
    """Rotation matrix about a unit axis (Rodrigues)."""
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def _R_to_quat(R):
    """wxyz quaternion from a rotation matrix (stable trace branch)."""
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-8:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    # w ~ 0: pick the dominant diagonal
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(1e-12, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", choices=["rigid", "hinge"], default="rigid")
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fovy", type=float, default=0.9, help="radians, as the renderer")
    ap.add_argument("--distance", type=float, default=0.0,
                    help="0 = fit automatically from the slab's size")
    ap.add_argument("--fill", type=float, default=0.6,
                    help="fraction of the frame the object spans when auto-fitting")
    ap.add_argument("--elevation", type=float, default=25.0, help="degrees")
    ap.add_argument("--azimuth", type=float, default=180.0,
                    help="degrees. 180 keeps BOTH hinge parts visible through the "
                         "hinge window after a 180-deg flip (measured: azimuth 0 "
                         "leaves the body fully occluded by the opening cover)")
    ap.add_argument("--body-size", default="0.20,0.15,0.02",
                    help="full body extents in metres, LxWxT; rigid default "
                         "overrides thickness to 0.006 unless set explicitly")
    ap.add_argument("--cover-thickness", type=float, default=0.004,
                    help="hinge variant: cover full thickness in metres")
    ap.add_argument("--turnover-deg", type=float, default=180.0)
    ap.add_argument("--turnover-axis", choices=["x", "y"], default="y",
                    help="in-plane axis the slab flips about")
    ap.add_argument("--turnover-window", default="0.15,0.55",
                    help="fractional window a,b of the turnover motion")
    ap.add_argument("--hinge-deg", type=float, default=120.0)
    ap.add_argument("--hinge-window", default="0.62,0.92")
    ap.add_argument("--overlap", action="store_true",
                    help="hinge opens DURING the turnover (simultaneous motion) "
                         "instead of after it")
    ap.add_argument("--translation", type=float, default=0.0,
                    help="whole-object travel in metres over the sequence")
    ap.add_argument("--translation-axis", choices=["x", "z"], default="x",
                    help="z = vertical lift with a fixed camera (the Gaussian-"
                         "duplication user report); x = the renderer's convention")
    ap.add_argument("--ground", action="store_true")
    ap.add_argument("--occluder", default=None, metavar="A,B",
                    help="sweep a foreground bar across the object during the "
                         "fractional window [A,B]: temporary occlusion. The bar "
                         "is NOT a part -- seg stays 255 there, so the oracle "
                         "union mask honestly excludes occluded object pixels, "
                         "and depth shows the bar, like a real hand passing "
                         "through the view")
    ap.add_argument("--occluder-mode", choices=["sweep", "dwell"], default="sweep",
                    help="dwell = pause at the object centre for the middle "
                         "third of the window, cutting the visible object in two")
    ap.add_argument("--faces", choices=["uniform", "distinct"], default="uniform",
                    help="uniform = one constant material, top and bottom faces "
                         "visually identical (verified: byte-identical colour "
                         "stats across a 180-deg flip). distinct = light top "
                         "half / dark-red bottom half of the SAME rigid link, "
                         "so the opposite face is visually distinguishable; "
                         "geometry and seg are unchanged")
    ap.add_argument("--depth-noise", type=float, default=0.0,
                    help="std = this * range^2, as the renderer")
    ap.add_argument("--depth-quant-mm", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if os.path.exists(os.path.join(args.out, "meta.json")):
        ap.error("output already contains a sequence; choose a new directory")
    body = np.array([float(x) for x in args.body_size.split(",")])
    if args.variant == "rigid" and args.body_size == ap.get_default("body_size"):
        body[2] = 0.006                       # a closed thin book, not a block
    render(args, body)


def render(args, body):
    import cv2
    import sapien
    from scripts.sim.render_partnet_sequence import look_at, add_depth_noise, _quat_to_R

    rng = np.random.default_rng(args.seed)
    scene = sapien.Scene()
    scene.set_timestep(1 / 60)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0.3, 0.5, -1], [2.2, 2.2, 2.2])
    scene.add_point_light([1.0, 1.0, 1.5], [12, 12, 12])

    hx, hy, hz = body / 2
    hc = args.cover_thickness / 2
    hinge_rad = np.radians(args.hinge_deg)

    # One articulation for both variants; the rigid one is a single root link.
    # Physics is never stepped -- poses and qpos are scripted, like the renderer.
    b = scene.create_articulation_builder()
    root = b.create_link_builder()
    root.set_name("body")
    if args.faces == "distinct":
        # Two half-thickness visuals on ONE rigid link: light top, dark-red
        # bottom. Same geometry, same entity (seg/mask unchanged); only the
        # appearance symmetry across a flip is broken.
        root.add_box_visual(sapien.Pose(p=[0, 0, hz / 2]),
                            half_size=[hx, hy, hz / 2], material=[0.72, 0.66, 0.55])
        root.add_box_visual(sapien.Pose(p=[0, 0, -hz / 2]),
                            half_size=[hx, hy, hz / 2], material=[0.55, 0.18, 0.14])
    else:
        root.add_box_visual(half_size=[hx, hy, hz], material=[0.72, 0.66, 0.55])
    root.add_box_collision(half_size=[hx, hy, hz])
    part_names = ["body"]
    if args.variant == "hinge":
        qz90 = _R_to_quat(_rot_about(np.array([0., 0., 1.]), np.pi / 2))
        cov = b.create_link_builder(root)
        cov.set_name("cover")
        cov.set_joint_name("spine")
        cov.add_box_visual(half_size=[hx, hy, hc], material=[0.45, 0.30, 0.22])
        cov.add_box_collision(half_size=[hx, hy, hc])
        # Joint frame at the spine (x = -hx edge, top of the body), axis = the
        # joint frame's x, rotated to lie along the body's y by the z-90 turn.
        cov.set_joint_properties("revolute", [[0.0, hinge_rad]],
                                 sapien.Pose(p=[-hx, 0, hz + hc], q=qz90),
                                 sapien.Pose(p=[-hx, 0, 0], q=qz90))
        part_names.append("cover")
    art = b.build(fix_root_link=False)
    art.set_name("thin_slab")
    links = art.get_links()
    link_of = {l.get_name(): l for l in links}
    ent_id = {l.get_name(): l.get_entity().per_scene_id for l in links}
    dof = art.dof

    # Camera: FIXED. Auto-distance from the largest extent the motion reaches
    # (an open cover doubles the x span).
    reach = np.linalg.norm(body) / 2 * (2.0 if args.variant == "hinge" else 1.0)
    if args.distance <= 0:
        args.distance = max(0.35, reach / (args.fill * np.tan(args.fovy / 2)))
    el, az = np.radians(args.elevation), np.radians(args.azimuth)
    eye = args.distance * np.array([np.cos(el) * np.cos(az),
                                    np.cos(el) * np.sin(az), np.sin(el)])
    if args.ground:
        gz = -float(np.linalg.norm(body) / 2) - 0.01
        gb = scene.create_actor_builder()
        gb.add_plane_collision(sapien.Pose(p=[0, 0, gz], q=[0.7071, 0, -0.7071, 0]))
        gb.add_plane_visual(sapien.Pose(p=[0, 0, gz], q=[0.7071, 0, -0.7071, 0]),
                            scale=[10, 10, 10], material=[0.55, 0.5, 0.45])
        gb.build_static(name="ground")
    cam = scene.add_camera("cam", args.width, args.height, args.fovy, 0.05, 10.0)
    cam.set_local_pose(look_at(eye, [0., 0., 0.]))
    K = cam.get_intrinsic_matrix()

    # Temporary occluder: a bar between camera and object, swept across the
    # view. Not a part; its pixels stay background in seg (mask excludes them)
    # while depth reports the bar -- a hand crossing a real capture.
    bar, bar_y = None, None
    if args.occluder:
        oa, ob = parse_window(args.occluder)
        span = 0.24
        if args.occluder_mode == "dwell":
            # in over the first third of the window, HOLD at the object centre
            # for the middle third (a hand gripping the middle), out over the
            # last third -- the visible object is two disjoint halves meanwhile
            third = (ob - oa) / 3
            bar_y = (-span / 2 + window_profile(args.frames, span / 2, oa, oa + third)
                     + window_profile(args.frames, span / 2, ob - third, ob))
        else:
            bar_y = -span / 2 + window_profile(args.frames, span, oa, ob)
        bb = scene.create_actor_builder()
        bb.add_box_visual(half_size=[0.02, 0.015, 0.25], material=[0.15, 0.15, 0.17])
        bb.add_box_collision(half_size=[0.02, 0.015, 0.25])
        bar = bb.build_kinematic(name="occluder")

    # Scripted trajectories. Turnover is root motion; the hinge is a joint.
    a, bnd = parse_window(args.turnover_window)
    theta = window_profile(args.frames, np.radians(args.turnover_deg), a, bnd)
    if args.variant == "hinge":
        ha, hb = (a, bnd) if args.overlap else parse_window(args.hinge_window)
        qh = window_profile(args.frames, hinge_rad, ha, hb)
        Q = qh[:, None]                                     # (T, 1)
    else:
        Q = np.zeros((args.frames, 0))
    axis = _axis_vector(args.turnover_axis)
    tvec = (np.array([0., 0., 1.]) if args.translation_axis == "z"
            else np.array([1., 0., 0.]))
    tx = args.translation * np.arange(args.frames) / max(1, args.frames - 1)

    for d in ("rgb", "depth", "seg"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    T = args.frames
    poses = np.zeros((T, len(part_names), 4, 4))
    cam_poses = np.zeros((T, 4, 4))
    mask_px = np.zeros((T, len(part_names)), dtype=np.int64)
    pixel_reprojection_errors = []

    for t in range(T):
        R = _rot_about(axis, theta[t])
        art.set_root_pose(sapien.Pose(p=tx[t] * tvec, q=_R_to_quat(R)))
        if bar is not None:
            bar.set_pose(sapien.Pose(p=eye * 0.5 + np.array([0., bar_y[t], 0.])))
        if dof:
            art.set_qpos(Q[t])
            if not np.allclose(art.get_qpos(), Q[t], atol=1e-6):
                raise RuntimeError("simulator joint state differs from scripted GT")
        scene.update_render()
        cam.take_picture()

        rgb = (np.clip(cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
        pos = cam.get_picture("Position")
        z = -pos[..., 2].astype(np.float64)
        z[z <= 1e-6] = 0.0
        valid = z > 1e-6
        yy, xx = np.where(valid)
        xyz_cv = pos[valid, :3] * np.array([1, -1, -1])
        uvz = xyz_cv @ K.T
        if len(uvz):
            uv = uvz[:, :2] / uvz[:, 2:]
            err = np.max(np.linalg.norm(uv - np.stack([xx + .5, yy + .5], axis=1), axis=1))
            pixel_reprojection_errors.append(float(err))
            if err > .1:
                raise RuntimeError(f"RGB/depth projection mismatch: {err} pixels")
        z = add_depth_noise(z, args.depth_noise, args.depth_quant_mm, rng)
        seg_ent = cam.get_picture("Segmentation")[..., 1]
        seg = np.full(seg_ent.shape, 255, dtype=np.uint8)
        for k, name in enumerate(part_names):
            m = seg_ent == ent_id[name]
            seg[m] = k
            mask_px[t, k] = int(m.sum())

        cv2.imwrite(f"{args.out}/rgb/{t:06d}.png", rgb[..., ::-1])
        cv2.imwrite(f"{args.out}/depth/{t:06d}.png",
                    np.clip(np.rint(z * 1000), 0, 65535).astype(np.uint16))
        cv2.imwrite(f"{args.out}/seg/{t:06d}.png", seg)

        E = cam.get_extrinsic_matrix()
        T_cam_world = np.eye(4)
        T_cam_world[:3, :4] = E
        cam_poses[t] = np.linalg.inv(T_cam_world)
        for k, name in enumerate(part_names):
            p = link_of[name].get_entity_pose()
            M = np.eye(4)
            M[:3, :3] = _quat_to_R(p.q)
            M[:3, 3] = p.p
            poses[t, k] = T_cam_world @ M

    union = mask_px.sum(axis=1)
    edge_on = int(np.argmin(union))
    np.savez_compressed(os.path.join(args.out, "poses.npz"),
                        T_cam_part=poses, joint_states=Q, source_joint_states=Q,
                        cam_poses=cam_poses,
                        timestamps=np.arange(T) / args.fps, mask_px=mask_px)
    json.dump({
        "model_dir": "procedural:thin_slab",
        "category": "ThinBook",
        "variant": args.variant,
        "parts": part_names,
        "link_groups": {n: [n] for n in part_names},
        "source_joint_names": ["spine"] if args.variant == "hinge" else [],
        "intrinsics": K.tolist(),
        "image_size": [args.height, args.width],
        "joints": ([{"name": "spine", "type": "revolute", "parent": "body",
                     "child": "cover", "limits": [0.0, float(np.radians(args.hinge_deg))]}]
                   if args.variant == "hinge" else []),
        "frames": T,
        "pixel_reprojection_max_px": max(pixel_reprojection_errors, default=0.0),
        "depth_noise": args.depth_noise,
        "depth_quant_mm": args.depth_quant_mm,
        "depth_storage_mm": 1.0,
        "fps": args.fps,
        "seed": args.seed,
        "motion": "turnover",
        "scripted_kinematics": True,
        "manipulators": False,
        "body_size_m": [float(x) for x in body],
        "cover_thickness_m": (args.cover_thickness if args.variant == "hinge" else None),
        "turnover_deg": args.turnover_deg,
        "turnover_axis": args.turnover_axis,
        "turnover_window": list(parse_window(args.turnover_window)),
        "hinge_deg": (args.hinge_deg if args.variant == "hinge" else None),
        "hinge_window": (list((parse_window(args.turnover_window) if args.overlap
                               else parse_window(args.hinge_window)))
                         if args.variant == "hinge" else None),
        "overlap": bool(args.overlap),
        "object_translation_m": args.translation,
        "object_translation_axis": args.translation_axis,
        "occluder_window": (list(parse_window(args.occluder)) if args.occluder else None),
        "occluder_mode": (args.occluder_mode if args.occluder else None),
        "faces": args.faces,
        "camera_azimuth_deg": float(args.azimuth),
        "camera_elevation_deg": float(args.elevation),
        "distance_m": args.distance,
        "edge_on_frame": edge_on,
        "union_px_min": int(union.min()),
        "union_px_max": int(union.max()),
        "gt_note": "T_cam_part / seg part indices are EVAL-ONLY; the tracker "
                   "gets rgb, depth and the binary union mask (seg != 255).",
    }, open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    print(f"[thin] {args.out}: {T} frames, variant {args.variant}, "
          f"turnover {args.turnover_deg:.0f} deg about {args.turnover_axis}")
    print(f"[thin] union mask px: max {union.max()} -> min {union.min()} "
          f"at frame {edge_on} (edge-on)")


if __name__ == "__main__":
    main()

"""Render an articulated PartNet-Mobility object to an RGB-D sequence with exact GT.

Why this exists: every failure diagnosed on RBO so far ends at "the articulation
signal was near the depth noise floor", and on real data there is no way to
separate a bad method from bad data.  Here the depth noise is a dial, the
articulation is scripted, and the per-part poses are exact.

Writes, per sequence:
    rgb/000000.png        uint8
    depth/000000.png      uint16 millimetres, 0 = invalid
    seg/000000.png        uint8 part index, 255 = background
    meta.json             intrinsics, part names, joint list, render settings
    poses.npz             T_cam_part (T,K,4,4), joint states (T,J), camera poses

The layout is read back by SapienReader, which mirrors RBOReader's interface, so
the tracking pipeline runs on it unchanged.

Assets: PartNet-Mobility (sapien-sim/PartNetMobility), simulator: SAPIEN 3.
"""

import argparse
import json
import os

import cv2
import numpy as np
import sapien


def look_at(eye, target, up=(0, 0, 1)):
    """Camera pose looking at a target, in SAPIEN's convention (+x forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    fwd = np.asarray(target, dtype=np.float64) - eye
    fwd /= np.linalg.norm(fwd)
    left = np.cross(np.asarray(up, dtype=np.float64), fwd)
    if np.linalg.norm(left) < 1e-6:
        left = np.array([0.0, 1.0, 0.0])
    left /= np.linalg.norm(left)
    up2 = np.cross(fwd, left)
    R = np.stack([fwd, left, up2], axis=1)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = eye
    return sapien.Pose(T)


def joint_trajectory(n, lo, hi, static_prefix=0.2, open_frac=0.5, mode="open"):
    """Hold still, articulate, hold still again.

    The static prefix matters: it is the phase where a correct method must refuse
    to split, and several detectors were only ever tested on data where the
    object was already moving.
    """
    if n < 4 or not 0 <= static_prefix < .5 or not 0 < open_frac <= 1:
        raise ValueError("need n>=4, 0<=static_prefix<0.5 and 0<open_frac<=1")
    if mode == "static":
        return np.full(n, lo, dtype=np.float64)
    if mode == "open_close":
        q = np.full(n, lo, dtype=np.float64)
        a = int(n * static_prefix)
        duration = min(max(2, int(n * open_frac / 2)), (n - 2 * a) // 2)
        if duration < 2:
            raise ValueError("too few moving frames for open_close")
        ramp = .5 - .5 * np.cos(np.linspace(0, np.pi, duration))
        q[a:a + duration] = lo + (hi - lo) * ramp
        c = n - a - duration
        q[a + duration:c] = hi
        q[c:n - a] = lo + (hi - lo) * ramp[::-1]
        return q
    if mode != "open":
        raise ValueError(f"unknown motion: {mode}")
    q = np.empty(n, dtype=np.float64)
    a = int(n * static_prefix)
    b = a + max(1, int(n * open_frac))
    q[:a] = lo
    ramp = np.linspace(0.0, 1.0, max(1, b - a))
    smooth = 0.5 - 0.5 * np.cos(np.pi * ramp)          # ease in/out
    q[a:b] = lo + (hi - lo) * smooth[: max(0, min(b, n) - a)]
    q[b:] = hi
    return q


def motion_groups(link_names, relations, moving):
    """Merge fixed and unexcited joints into observable rigid motion groups."""
    parent = {name: name for name in link_names}
    def root(name):
        while parent[name] != name:
            name = parent[name]
        return name
    for name, a, b in relations:
        if name not in moving:
            parent[root(b)] = root(a)
    groups = {}
    for name in link_names:
        groups.setdefault(root(name), []).append(name)
    return groups


def add_depth_noise(z, sigma_rel, quant_mm, rng):
    """Structured-light-like noise: std grows with the square of range."""
    out = z.copy()
    m = z > 0
    if sigma_rel > 0:
        out[m] += rng.normal(0.0, sigma_rel * z[m] ** 2)
    if quant_mm > 0:
        out[m] = np.round(out[m] * 1000.0 / quant_mm) * quant_mm / 1000.0
    out[out < 0] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True, help="a PartNet-Mobility model directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fovy", type=float, default=0.9)
    ap.add_argument("--distance", type=float, default=0.0,
                    help="0 = fit automatically from the object's size")
    ap.add_argument("--fill", type=float, default=0.75,
                    help="fraction of the frame the object should span when auto-fitting")
    ap.add_argument("--object-size", type=float, default=0.35,
                    help="largest object dimension in metres. PartNet-Mobility "
                         "models are unit-normalised, not metric: a scissors "
                         "model is 1.9 m across as shipped, which put the scene "
                         "outside the tracker's max_depth and silently discarded "
                         "every 3D point.")
    ap.add_argument("--ground", action="store_true",
                    help="add a ground plane, so depth is not mostly empty")
    ap.add_argument("--elevation", type=float, default=25.0, help="degrees")
    ap.add_argument("--auto-view", action="store_true", default=True,
                    help="search viewpoints and keep the one where both parts stay "
                         "visible and the articulation is most legible")
    ap.add_argument("--no-auto-view", dest="auto_view", action="store_false")
    ap.add_argument("--view-azimuths", type=int, default=16)
    ap.add_argument("--view-elevations", default="15,30,45")
    ap.add_argument("--camera-orbit", type=float, default=0.0,
                    help="degrees of camera travel over the sequence (0 = static)")
    ap.add_argument("--camera-azimuth", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--motion-parts", action="store_true",
                    help="GT parts are rigid motion groups, merging fixed/unexcited joints")
    ap.add_argument("--active-joints", default=None,
                    help="comma-separated movable joint names; others stay at their start state")
    ap.add_argument("--static-prefix", type=float, default=0.2,
                    help="fraction of the sequence with no articulation")
    ap.add_argument("--open-frac", type=float, default=0.5)
    ap.add_argument("--motion", choices=["open", "open_close", "static"], default="open")
    ap.add_argument("--joint-range", type=float, default=1.0,
                    help="fraction of each joint's limit range to traverse")
    ap.add_argument("--start-frac", type=float, default=0.25,
                    help="where in the joint range the sequence starts. Starting "
                         "fully closed hides the moving part at frame 0, so no "
                         "method can put keypoints on it; 0.25 keeps both parts "
                         "visible from the outset.")
    ap.add_argument("--default-range-deg", type=float, default=90.0,
                    help="range to use for joints with unbounded limits")
    ap.add_argument("--depth-noise", type=float, default=0.0,
                    help="std = this * range^2 (0.0015 is roughly an Xtion at 1 m)")
    ap.add_argument("--depth-quant-mm", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 0 <= args.start_frac <= 1 or not 0 <= args.joint_range <= 1 or args.fps <= 0:
        ap.error("start-frac and joint-range must be in [0,1]; fps must be positive")
    if os.path.exists(os.path.join(args.out, "meta.json")):
        ap.error("output already contains a sequence; choose a new directory")

    rng = np.random.default_rng(args.seed)
    scene = sapien.Scene()
    scene.set_timestep(1 / 60)
    scene.set_ambient_light([0.4, 0.4, 0.4])
    scene.add_directional_light([0.3, 0.5, -1], [2.2, 2.2, 2.2])
    scene.add_point_light([1.0, 1.0, 1.5], [12, 12, 12])

    bb0 = json.load(open(os.path.join(args.model_dir, "bounding_box.json")))
    extent0 = float(np.max(np.array(bb0["max"]) - np.array(bb0["min"])))
    scale = args.object_size / max(extent0, 1e-6)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    loader.scale = scale
    art = loader.load(os.path.join(args.model_dir, "mobility.urdf"))
    print(f"[sim] model extent {extent0:.2f} (normalised) -> scale {scale:.3f} "
          f"for a {args.object_size:.2f} m object")

    links = art.get_links()
    part_names = [l.get_name() for l in links]
    ent_id = {l.get_name(): l.get_entity().per_scene_id for l in links}
    joints = [j for j in art.get_joints() if j.get_dof() > 0]
    if not joints:
        raise SystemExit("model has no movable joint")

    # Fit the initial asset bounding sphere. The suite validator separately
    # checks every rendered frame for clipping throughout articulation/orbit.
    # PartNet-Mobility ships the model's bounding box; use it rather than probing
    # collision shapes, whose bounds came back empty and collapsed the camera
    # distance to the floor value.
    bmin = np.array(bb0["min"], dtype=np.float64) * scale
    bmax = np.array(bb0["max"], dtype=np.float64) * scale
    root = art.get_root_pose().p
    centre = root + 0.5 * (bmin + bmax)
    radius = float(np.linalg.norm(bmax - bmin) / 2)
    if args.distance <= 0:
        args.distance = max(0.35, radius / (args.fill * np.tan(args.fovy / 2)))
    print(f"[sim] object radius {radius:.3f} m -> camera distance {args.distance:.2f} m")

    if args.ground:
        gz = float(centre[2] - radius) - 0.01
        b = scene.create_actor_builder()
        b.add_plane_collision(sapien.Pose(p=[0, 0, gz], q=[0.7071, 0, -0.7071, 0]))
        b.add_plane_visual(sapien.Pose(p=[0, 0, gz], q=[0.7071, 0, -0.7071, 0]),
                           scale=[10, 10, 10], material=[0.55, 0.5, 0.45])
        b.build_static(name="ground")

    cam = scene.add_camera("cam", args.width, args.height, args.fovy, 0.05, 10.0)
    K = cam.get_intrinsic_matrix()

    def _render_state(q, az, el):
        """Segmentation of one joint state from one viewpoint."""
        art.set_qpos(q)
        e = np.radians(el)
        eye = centre + args.distance * np.array(
            [np.cos(e) * np.cos(np.radians(az)), np.cos(e) * np.sin(np.radians(az)),
             np.sin(e)])
        cam.set_local_pose(look_at(eye, centre))
        scene.update_render()
        cam.take_picture()
        seg_ent = cam.get_picture("Segmentation")[..., 1]
        out = np.full(seg_ent.shape, -1, dtype=np.int16)
        for k, name in enumerate(part_names):
            out[seg_ent == ent_id[name]] = k
        return out

    q_lo = np.array([j.get_limits()[0][0] for j in joints], dtype=np.float64)
    q_hi = np.array([j.get_limits()[0][1] for j in joints], dtype=np.float64)
    for i, j in enumerate(joints):
        if not np.isfinite(q_lo[i]) or not np.isfinite(q_hi[i]):
            q_lo[i], q_hi[i] = 0.0, np.radians(args.default_range_deg)
    span_q = q_hi - q_lo
    q_a = q_lo + span_q * args.start_frac
    q_b = q_a + span_q * args.joint_range * (1.0 - args.start_frac)
    active = {j.get_name() for j in joints} if args.active_joints is None else \
        set(filter(None, args.active_joints.split(",")))
    unknown = active - {j.get_name() for j in joints}
    if unknown:
        ap.error(f"unknown active joints: {sorted(unknown)}")
    for k, j in enumerate(joints):
        if j.get_name() not in active or args.motion == "static":
            q_b[k] = q_a[k]

    if args.auto_view:
        # A viewpoint is only useful if BOTH parts stay visible across the whole
        # articulation and the motion is legible from it -- an edge-on door or a
        # laptop shot from behind its lid gives a mask a few pixels tall, which no
        # amount of tracking can recover.
        els = [float(x) for x in args.view_elevations.split(",")]
        best = None
        for el in els:
            for az in np.linspace(0, 360, args.view_azimuths, endpoint=False):
                sa, sb = _render_state(q_a, az, el), _render_state(q_b, az, el)
                px = [min(int((sa == k).sum()), int((sb == k).sum()))
                      for k in range(len(part_names))]
                px = [p for p, n in zip(px, part_names)]
                vis = [p for p in px if p > 200]
                if len(vis) < 2:
                    continue
                # how far the parts move relative to each other in the image
                cen = []
                for st in (sa, sb):
                    cs = []
                    for k in range(len(part_names)):
                        ys, xs = np.where(st == k)
                        cs.append(np.array([xs.mean(), ys.mean()]) if xs.size > 20
                                  else None)
                    cen.append(cs)
                rel = 0.0
                for k in range(len(part_names)):
                    for l in range(k + 1, len(part_names)):
                        if all(cen[s][m] is not None for s in (0, 1) for m in (k, l)):
                            d0 = np.linalg.norm(cen[0][k] - cen[0][l])
                            d1 = np.linalg.norm(cen[1][k] - cen[1][l])
                            rel = max(rel, abs(d1 - d0))
                score = min(vis) * (1.0 + rel / 50.0)
                if best is None or score > best[0]:
                    best = (score, az, el, sorted(px)[-2:], rel)
        if best is not None:
            _, args.camera_azimuth, args.elevation, px2, rel = best
            print(f"[sim] auto-view: azimuth {args.camera_azimuth:.0f} deg, "
                  f"elevation {args.elevation:.0f} deg  "
                  f"(smallest two parts {px2} px, relative motion {rel:.0f} px)")
        else:
            args.camera_azimuth = 0.0
            print("[sim] auto-view found no pose with two visible parts; using default")

    # scripted joint trajectories
    Q = []
    for j in joints:
        lo, hi = j.get_limits()[0]
        if not np.isfinite(lo) or not np.isfinite(hi):
            # continuous joints (globes, wheels) report infinite limits
            lo, hi = 0.0, np.radians(args.default_range_deg)
            print(f"[sim] joint {j.get_name()} is unbounded; using "
                  f"0..{args.default_range_deg:.0f} deg")
        span = hi - lo
        lo = lo + span * args.start_frac
        hi = lo + span * args.joint_range * (1.0 - args.start_frac)
        Q.append(joint_trajectory(args.frames, lo, hi, args.static_prefix,
                                  args.open_frac, args.motion if j.get_name() in active else "static"))
    Q = np.stack(Q, axis=1)                                   # (T, J)

    moving = {j.get_name() for k, j in enumerate(joints) if np.ptp(Q[:, k]) > 1e-7}
    relations = [(j.get_name(), j.get_parent_link().get_name(), j.get_child_link().get_name())
                 for j in art.get_joints() if j.get_parent_link() is not None]
    groups = motion_groups(part_names, relations, moving) if args.motion_parts else \
        {name: [name] for name in part_names}
    output_names = list(groups)
    group_of = {link: name for name, members in groups.items() for link in members}
    output_index = {link: output_names.index(group_of[link]) for link in part_names}
    output_links = [links[part_names.index(name)] for name in output_names]
    output_joint_indices = [k for k, j in enumerate(joints)
                            if not args.motion_parts or j.get_name() in moving]

    for d in ("rgb", "depth", "seg"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)

    poses = np.zeros((args.frames, len(output_links), 4, 4))
    cam_poses = np.zeros((args.frames, 4, 4))
    pixel_reprojection_errors = []

    for t in range(args.frames):
        art.set_qpos(Q[t])
        if not np.allclose(art.get_qpos(), Q[t], atol=1e-6):
            raise RuntimeError("simulator joint state differs from scripted GT")
        ang = np.radians(args.camera_azimuth) + \
            np.radians(args.camera_orbit) * (t / max(1, args.frames - 1))
        el = np.radians(args.elevation)
        eye = centre + args.distance * np.array(
            [np.cos(el) * np.cos(ang), np.cos(el) * np.sin(ang), np.sin(el)])
        cam.set_local_pose(look_at(eye, centre))
        scene.update_render()
        cam.take_picture()

        rgb = (np.clip(cam.get_picture("Color")[..., :3], 0, 1) * 255).astype(np.uint8)
        pos = cam.get_picture("Position")
        z = -pos[..., 2].astype(np.float64)          # camera looks down -z
        z[z <= 1e-6] = 0.0
        # Position uses OpenGL camera coordinates; verify CV projection before noise.
        valid = z > 1e-6
        yy, xx = np.where(valid)
        xyz_cv = pos[valid, :3] * np.array([1, -1, -1])
        uvz = xyz_cv @ K.T
        uv = uvz[:, :2] / uvz[:, 2:]
        if len(uv):
            error = np.max(np.linalg.norm(uv - np.stack([xx + .5, yy + .5], axis=1), axis=1))
            pixel_reprojection_errors.append(float(error))
            if error > .1:
                raise RuntimeError(f"RGB/depth projection mismatch: {error} pixels")
        z = add_depth_noise(z, args.depth_noise, args.depth_quant_mm, rng)
        seg_ent = cam.get_picture("Segmentation")[..., 1]

        seg = np.full(seg_ent.shape, 255, dtype=np.uint8)
        for k, name in enumerate(part_names):
            seg[seg_ent == ent_id[name]] = output_index[name]

        cv2.imwrite(f"{args.out}/rgb/{t:06d}.png", rgb[..., ::-1])
        cv2.imwrite(f"{args.out}/depth/{t:06d}.png",
                    np.clip(np.rint(z * 1000), 0, 65535).astype(np.uint16))
        cv2.imwrite(f"{args.out}/seg/{t:06d}.png", seg)

        E = cam.get_extrinsic_matrix()               # world -> camera, OpenCV
        T_cam_world = np.eye(4)
        T_cam_world[:3, :4] = E
        cam_poses[t] = np.linalg.inv(T_cam_world)
        for k, l in enumerate(output_links):
            p = l.get_pose()
            T = np.eye(4)
            T[:3, :3] = _quat_to_R(p.q)
            T[:3, 3] = p.p
            poses[t, k] = T_cam_world @ T

    np.savez_compressed(os.path.join(args.out, "poses.npz"),
                        T_cam_part=poses, joint_states=Q[:, output_joint_indices],
                        source_joint_states=Q, cam_poses=cam_poses,
                        timestamps=np.arange(args.frames) / args.fps)
    json.dump({
        "model_dir": args.model_dir,
        "category": json.load(open(os.path.join(args.model_dir, "meta.json"))).get("model_cat"),
        "parts": output_names,
        "link_groups": groups,
        "source_joint_names": [j.get_name() for j in joints],
        "intrinsics": K.tolist(),
        "image_size": [args.height, args.width],
        "joints": [{"name": j.get_name(), "type": str(j.type),
                    "parent": group_of[j.get_parent_link().get_name()],
                    "child": group_of[j.get_child_link().get_name()],
                    "limits": [float(x) for x in j.get_limits()[0]]}
                   for k, j in enumerate(joints) if k in output_joint_indices],
        "frames": args.frames,
        "pixel_reprojection_max_px": max(pixel_reprojection_errors, default=0.0),
        "depth_noise": args.depth_noise,
        "depth_quant_mm": args.depth_quant_mm,
        "depth_storage_mm": 1.0,
        "fps": args.fps,
        "seed": args.seed,
        "motion": args.motion,
        "scripted_kinematics": True,
        "manipulators": False,
        "object_scale": scale,
        "distance_m": args.distance,
        "start_frac": args.start_frac,
        "joint_range": args.joint_range,
        "open_frac": args.open_frac,
        "camera_orbit_deg": args.camera_orbit,
        "camera_azimuth_deg": float(args.camera_azimuth),
        "camera_elevation_deg": float(args.elevation),
        "static_prefix": args.static_prefix,
    }, open(os.path.join(args.out, "meta.json"), "w"), indent=2)

    print(f"[sim] {args.out}: {args.frames} frames, parts {output_names}")
    print(f"[sim] joint travel: " + ", ".join(
        f"{j.get_name()}={np.degrees(Q[:,i].max()-Q[:,i].min()):.0f}deg"
        if "revolute" in str(j.type) else
        f"{j.get_name()}={1000*(Q[:,i].max()-Q[:,i].min()):.0f}mm"
        for i, j in enumerate(joints)))


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


if __name__ == "__main__":
    main()

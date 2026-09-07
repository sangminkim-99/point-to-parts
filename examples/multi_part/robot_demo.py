"""Step 7: command the object we discovered, with a robot that is holding it.

The claim this exists to make is that the kinematics we recovered are good
enough to act on -- not that we can simulate contact. We have surfaces from a
single viewpoint and no masses, so a physics engine would be asserting things
we never measured. Instead:

  * one part is FIXED, either because a hand is holding it or because it is
    standing on a table. A gizmo moves that part, and everything else follows
    the joints we found.
  * the robot is assumed to have grasped a point on a moving part. Reaching
    that point as the joint sweeps its observed range is an IK problem, and
    doing it smoothly over the whole sweep is a small trajectory optimisation.

Nothing here is claimed about contact, friction or mass. What is claimed is
that the axis, the limits and the link geometry we recovered are enough to
plan against.

    python -m examples.multi_part.robot_demo results/urdf/ikeasmall02.urdf
    python -m examples.multi_part.robot_demo <urdf> --robot ur5 --no-serve
"""
import argparse
import os

import numpy as np

from examples.multi_part.urdf_view import PAL, read_urdf, link_poses

ROBOTS = {
    "panda": ("robot_descriptions.panda_description", "URDF_PATH"),
    "ur5": ("robot_descriptions.ur5_description", "URDF_PATH"),
    "yumi": ("robot_descriptions.yumi_description", "URDF_PATH"),
}


def load_robot(name):
    """A robot URDF from robot_descriptions, as a pyroki Robot."""
    import importlib
    import yourdfpy
    import pyroki
    mod, attr = ROBOTS.get(name, ROBOTS["panda"])
    path = getattr(importlib.import_module(mod), attr)
    urdf = yourdfpy.URDF.load(path)
    return pyroki.Robot.from_urdf(urdf), urdf


def grasp_point(geoms, joints, child, frac=0.7):
    """Where the robot holds: on the moving part, where the lever is longest.

    For a revolute joint that means AWAY from the hinge -- you open a door by
    its far edge, not by pushing along the hinge line. The first version
    pushed along the joint axis for both types, which for a revolute runs
    along the hinge and put the point inside the body the lid closes onto:
    cardboardbox01 came out with 23 mm of reach error because the arm was
    aiming into the box. For a prismatic joint the axis IS the direction the
    part comes out, so there the push along it is right.

    Still a stand-in for the interaction point proper -- a handle, a lid edge
    -- which is step 6 and not yet recovered.
    """
    m = geoms.get(child)
    if m is None:
        return None
    v = np.asarray(m.vertices)
    j = next((j for j in joints if j["child"] == child), None)
    if j is None:
        return v.mean(0)
    a = np.asarray(j["axis"], float)
    a = a / max(np.linalg.norm(a), 1e-9)
    if j["type"] == "revolute":
        p0 = np.asarray(j["origin"], float)
        r = v - p0
        lever = np.linalg.norm(r - np.outer(r @ a, a), axis=1)
        pick = np.argsort(-lever)[:max(20, len(v) // 50)]
        return v[pick].mean(0)
    c = v.mean(0)
    return c + a * frac * 0.5 * float(np.linalg.norm(v.max(0) - v.min(0)))


def object_spheres(geoms, rest, centre, joints, root, q, k, n_pts, radius,
                   skip=None):
    """The object at this joint value, as spheres, for collision.

    A point cloud rather than the meshes. The meshes are Poisson shells fitted
    to one viewpoint -- they are not watertight and they invent surface -- so
    a mesh collider would be asserting geometry we never measured. The points
    ARE the measurement, farthest-point sampled so that a few hundred of them
    still cover the object, and a sphere of half the spacing closes the gaps.
    """
    import jax.numpy as jnp
    import open3d as o3d
    import pyroki.collision as pc
    full = np.zeros(len(joints)); full[k] = q
    pose = link_poses(joints, root, full)
    out = []
    for n, v in rest.items():
        if n == skip:
            # The part being held is not an obstacle. A gripper closes AROUND
            # what it grasps, so keeping the whole object in the collision set
            # forbids the very contact the demo is about: on cardboardbox01
            # that showed up as 21.7 mm of reach error, the arm held off the
            # surface it was supposed to be holding. The rest of the object --
            # the frame the door swings in -- stays an obstacle, which is what
            # makes re-grasping a real question.
            continue
        T = pose.get(n, np.eye(4))
        out.append((v - centre) @ T[:3, :3].T + T[:3, 3])
    if not out:
        return np.zeros((0, 3)), None
    P = np.concatenate(out)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    if len(P) > n_pts:
        pcd = pcd.farthest_point_down_sample(n_pts)
    Q = np.asarray(pcd.points)
    return Q, pc.Sphere.from_center_and_radius(
        jnp.asarray(Q), jnp.full(len(Q), radius))


def solve_path(robot, targets, link_index, smooth=2.0, rest_w=0.01,
               worlds=None, robot_coll=None, margin=0.004, coll_w=8.0):
    """One configuration per target, smooth along the way.

    Each waypoint is an IK problem; the smoothness term between neighbours is
    what makes it a trajectory rather than a sequence of unrelated poses. The
    object is a different obstacle at every waypoint, because it is the thing
    that moves, so each waypoint carries its own collision term.
    """
    import jax.numpy as jnp
    import jaxls
    import jaxlie
    import pyroki.costs as pkc

    n = len(targets)
    # One cost per waypoint. Handing pose_cost a batch of targets makes jaxls
    # try to batch the robot's own arrays with them, which do not share the
    # waypoint axis; per-waypoint costs sidestep that and 24 of them is
    # nothing to solve.
    lo = jnp.asarray(robot.joints.lower_limits)
    hi = jnp.asarray(robot.joints.upper_limits)
    rest = 0.5 * (lo + hi)      # a neutral posture to be pulled back to
    vars_ = [robot.joint_var_cls(i) for i in range(n)]
    costs = []
    for i, (v, t) in enumerate(zip(vars_, targets)):
        costs.append(pkc.pose_cost(
            robot, v,
            jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3.identity(), jnp.asarray(t, jnp.float32)),
            jnp.asarray(link_index), 5.0, 0.0))
        costs.append(pkc.limit_cost(robot, v, 100.0))
        costs.append(pkc.rest_cost(v, rest, rest_w))
        if worlds is not None and robot_coll is not None:
            costs.append(pkc.world_collision_cost(
                robot, robot_coll, v, worlds[i], margin, coll_w))
        if i:
            costs.append(pkc.smoothness_cost(v, vars_[i - 1], smooth))
    sol = (jaxls.LeastSquaresProblem(costs, vars_).analyze()
           .solve(verbose=False))
    return np.stack([np.asarray(sol[v]) for v in vars_])



def _link_clouds(urdf, stride=3):
    """Vertices per geometry node, in that node's own frame."""
    out = []
    for node in urdf.scene.graph.nodes_geometry:
        name = urdf.scene.graph[node][1]
        m = urdf.scene.geometry.get(name)
        if m is None or not hasattr(m, "vertices"):
            continue
        out.append((node, np.asarray(m.vertices, np.float64)[::stride]))
    return out


def render_sweep(args, robot, urdf, base, cfgs, qs, k, joints, root,
                 geoms, rest, centre, radius, grip, ee_i, err=None):
    """The sweep as a video: the object articulating and the arm following.

    Drawn by projecting and splatting, the same way urdf_view does, because a
    headless GL context on this machine returns a frame that is not what the
    viewer shows.
    """
    import cv2
    W, H = args.size
    clouds = _link_clouds(urdf)
    # A viewpoint across the approach, not along it. The base now stands on
    # the side the camera saw, so the recording camera's own direction has the
    # arm directly in front of the object.
    tgt = 0.5 * base
    v = base / max(np.linalg.norm(base), 1e-6)
    side = np.cross(v, [0, 1, 0])
    side = side / max(np.linalg.norm(side), 1e-6)
    eye = tgt + side * 1.35 + np.array([0.0, -0.75, 0.0]) - v * 0.35
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, -1, 0]); right /= np.linalg.norm(right)
    Rc = np.stack([right, np.cross(right, fwd), fwd])
    f = 1.25 * max(W, H)

    def project(v):
        c = Rc @ (v - eye).T
        z = c[2]
        ok = z > 1e-3
        return (np.stack([f * c[0][ok] / z[ok] + W / 2,
                          f * c[1][ok] / z[ok] + H / 2], 1), z[ok])

    def splat(img, P, D, colour):
        keep = (P[:, 0] > -10) & (P[:, 0] < W + 10) & \
               (P[:, 1] > -10) & (P[:, 1] < H + 10)
        return P[keep], D[keep], np.full(int(keep.sum()), colour)

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (W, H))
    order = list(range(len(qs))) + list(range(len(qs) - 1, -1, -1))
    for step, i in enumerate(order):
        img = np.full((H, W, 3), (24, 27, 31), np.uint8)
        P, D, C = [], [], []
        # the object, at this joint value
        full = np.zeros(len(joints)); full[k] = qs[i]
        pose = link_poses(joints, root, full)
        for li, n in enumerate(geoms):
            T = pose.get(n, np.eye(4))
            v = (rest[n] - centre) @ T[:3, :3].T + T[:3, 3]
            p, d = project(v)
            a, b, c = splat(img, p, d, li)
            P.append(a); D.append(b); C.append(c)
        # the arm, at the configuration solved for it
        urdf.update_cfg(np.asarray(cfgs[i]))
        for node, verts in clouds:
            T = urdf.scene.graph[node][0]
            v = verts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3] + base
            p, d = project(v)
            a, b, c = splat(img, p, d, 99)
            P.append(a); D.append(b); C.append(c)
        P = np.concatenate(P); D = np.concatenate(D); C = np.concatenate(C)
        o = np.argsort(-D)
        P, D, C = P[o], D[o], C[o]
        lo, hi = D.min(), D.max()
        sh = 1.15 - 0.5 * (D - lo) / max(hi - lo, 1e-6)
        for (x, y), s_, ci in zip(P.astype(int), sh, C):
            col = (0.72, 0.74, 0.78) if ci == 99 else PAL[int(ci) % len(PAL)]
            b_, g_, r_ = col[::-1]
            cv2.circle(img, (x, y), 2, (int(255 * min(b_ * s_, 1)),
                                        int(255 * min(g_ * s_, 1)),
                                        int(255 * min(r_ * s_, 1))), -1)
        # the point the robot is holding
        p, d = project(grip[i][None])
        if len(p):
            cv2.circle(img, tuple(p[0].astype(int)), 6, (20, 20, 20), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p[0].astype(int)), 4, (120, 240, 250), -1, cv2.LINE_AA)
        j = joints[k]
        unit = ("deg", np.degrees(qs[i])) if j["type"] == "revolute" \
            else ("mm", qs[i] * 1000)
        cv2.rectangle(img, (0, H - 34), (W, H), (16, 18, 21), -1)
        cap = (f"{j['name']}  {j['type']}   q = {unit[1]:+.1f} {unit[0]}"
               f"   |  root {root} held fixed"
               + (f"   |  reach error {1000 * err[i]:.1f} mm"
                  if err is not None else ""))
        cv2.putText(img, cap, (12, H - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (228, 228, 228), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    print(f"[robot] wrote {args.out} ({len(order)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf", help="the object URDF written by --urdf")
    ap.add_argument("--robot", default="panda", choices=sorted(ROBOTS))
    ap.add_argument("--joint", type=int, default=None,
                    help="which joint to drive; default is the widest one")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--no-serve", action="store_true",
                    help="solve and report, without starting the viewer")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--out", default=None,
                    help="write the sweep as an mp4 instead of serving it")
    ap.add_argument("--size", type=int, nargs=2, default=[960, 720])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--coll-points", type=int, default=400,
                    help="farthest-point samples of the object, per waypoint")
    ap.add_argument("--coll-radius", type=float, default=0.012)
    ap.add_argument("--reach", type=float, nargs=2, default=[0.32, 0.72],
                    help="the arm's comfortable annulus, metres from its base")
    ap.add_argument("--try-bases", type=int, default=4)
    ap.add_argument("--coll-margin", type=float, default=0.004,
                    help="clearance demanded of the obstacles. Small on "
                         "purpose: grasping an edge IS approaching the body "
                         "it sits on, and a generous berth simply forbids the "
                         "contact the demo exists to show. The term is here to "
                         "stop the arm passing THROUGH what we measured.")
    ap.add_argument("--no-collision", action="store_true")
    args = ap.parse_args()

    import open3d as o3d
    links, joints, root = read_urdf(args.urdf)
    geoms = {}
    for n, path in links.items():
        if path and os.path.exists(path):
            m = o3d.io.read_triangle_mesh(path)
            m.compute_vertex_normals()
            geoms[n] = m
    if not joints or not geoms:
        print("[robot] the URDF has no joints or no meshes")
        return
    rest = {n: np.asarray(m.vertices).copy() for n, m in geoms.items()}

    k = args.joint
    if k is None:
        k = int(np.argmax([j["upper"] - j["lower"] for j in joints]))
    j = joints[k]
    span = j["upper"] - j["lower"]
    unit = "deg" if j["type"] == "revolute" else "mm"
    print(f"[robot] object: {len(geoms)} links, {len(joints)} joints, root {root}")
    print(f"[robot] driving {j['name']}: {j['type']} {j['parent']} -> "
          f"{j['child']}, observed range "
          f"{(np.degrees(span) if unit == 'deg' else span * 1000):.1f} {unit}")

    # the object sits where the camera saw it; put its root at the origin so a
    # robot in its own base frame can be placed beside it
    allv = np.vstack([v for v in rest.values()])
    centre = allv.mean(0)
    radius = float(np.linalg.norm(allv - centre, axis=1).max())

    qs = np.linspace(j["lower"], j["upper"], args.steps)
    grip = []
    for q in qs:
        full = np.zeros(len(joints)); full[k] = q
        pose = link_poses(joints, root, full)
        T = pose.get(j["child"], np.eye(4))
        g = grasp_point(geoms, joints, j["child"])
        grip.append(T[:3, :3] @ g + T[:3, 3] - centre)
    grip = np.asarray(grip)
    travel = float(np.linalg.norm(grip[-1] - grip[0]))
    print(f"[robot] the grasp point travels {travel * 1000:.0f} mm over that range")

    robot, robot_urdf = load_robot(args.robot)
    names = list(robot.links.names)
    # the tool frame, not whichever link happens to be listed last
    ee = next((n for n in ("panda_hand_tcp", "panda_hand", "tool0", "ee_link",
                           "gripper_r_finger_l") if n in names), names[-1])
    ee_i = names.index(ee)
    print(f"[robot] {args.robot}: {robot.joints.num_actuated_joints} actuated "
          f"joints, end effector '{ee}'")

    # Stand the robot where the whole sweep is inside its envelope. Too far
    # and the arm is at full stretch, which shows up as tens of millimetres of
    # reach error that say nothing about the kinematics we recovered.
    # Where to stand. The geometry is in camera coordinates, so the side we
    # have any surface for -- and the side the parts open toward -- is the
    # camera's side, which is -z. That fixes the direction; what is left is
    # how far back and how far across, and those are chosen by keeping the
    # whole sweep inside the arm's comfortable annulus rather than by trying
    # trajectories, which would be a solve per candidate.
    lo, hi = args.reach
    cands = []
    for d in np.arange(0.30, 0.95, 0.05):
        for lat in (-0.25, -0.12, 0.0, 0.12, 0.25):
            for dz in (-0.30, -0.15, 0.0):
                b = np.array([lat, dz, -d])
                r = np.linalg.norm(grip - b, axis=1)
                if r.min() < lo or r.max() > hi:
                    continue
                # the middle of the annulus, not merely inside it: a base
                # that only just satisfies the bound puts the arm at a stretch
                # for half the sweep
                cands.append((-abs(r.mean() - 0.5 * (lo + hi)), b))
    if not cands:
        cands = [(0.0, np.array([0.0, -0.15, -0.6]))]
    cands.sort(key=lambda x: -x[0])
    # the object as points, once per waypoint, in the world frame
    import jax.numpy as jnp
    import pyroki.collision as pc
    coll = None if args.no_collision else pc.RobotCollision.from_urdf(robot_urdf)
    clouds = [object_spheres(geoms, rest, centre, joints, root, q, k,
                             args.coll_points, args.coll_radius,
                             skip=j["child"])[0] for q in qs]
    print(f"[robot] obstacles: {len(clouds[0])} spheres of "
          f"{1000 * args.coll_radius:.0f} mm, re-posed at every waypoint "
          f"(the held part {j['child']} is excluded)")

    best = None
    for _, b in cands[:args.try_bases]:
        worlds = None if coll is None else [
            pc.Sphere.from_center_and_radius(
                jnp.asarray(Q - b), jnp.full(len(Q), args.coll_radius))
            for Q in clouds]
        c = solve_path(robot, grip - b, ee_i, worlds=worlds,
                       robot_coll=coll, margin=args.coll_margin)
        f = np.asarray(robot.forward_kinematics(c))[:, ee_i, 4:7] + b
        e = float(np.linalg.norm(f - grip, axis=1).max())
        if best is None or e < best[0]:
            best = (e, b, c)
    _, base, cfgs = best
    print(f"[robot] stood the base at {np.round(base, 2)} m "
          f"(reach {np.linalg.norm(grip - base, axis=1).min():.2f}"
          f"-{np.linalg.norm(grip - base, axis=1).max():.2f} m)")
    fk = np.asarray(robot.forward_kinematics(cfgs))[:, ee_i, 4:7] + base
    err = np.linalg.norm(fk - grip, axis=1)
    print(f"[robot] IK over {args.steps} waypoints: reach error median "
          f"{1000 * np.median(err):.1f} mm, worst {1000 * err.max():.1f} mm")
    step = np.abs(np.diff(cfgs, axis=0)).max(axis=1)
    print(f"[robot] largest joint step between waypoints: "
          f"{np.degrees(step.max()):.1f} deg")
    if coll is not None:
        d = [float(np.asarray(coll.compute_world_collision_distance(
                robot, jnp.asarray(cfgs[i]),
                pc.Sphere.from_center_and_radius(
                    jnp.asarray(clouds[i] - base),
                    jnp.full(len(clouds[i]), args.coll_radius)))).min())
             for i in range(len(cfgs))]
        d = np.asarray(d)
        print(f"[robot] closest approach to the object: {1000 * d.min():+.0f} mm "
              f"(negative = through it), "
              f"{int((d < 0).sum())} of {len(d)} waypoints in contact")
        if err.max() > 0.01 or (d < 0).any():
            print("[robot] NOT feasible as posed. The arm cannot hold that "
                  "point over that range without going through the rest of "
                  "the object -- which is a real answer, not a solver "
                  "failure: a lid opened 18 degrees has its far edge resting "
                  "on the body it closes onto, and no gripper reaches there "
                  "either. It needs a grasp chosen for clearance, or the "
                  "object opened further before the robot takes over.")
    if args.out:
        render_sweep(args, robot, robot_urdf, base, cfgs, qs, k,
                     joints, root, geoms, rest, centre, radius, grip, ee_i,
                     err)
        return
    if args.no_serve:
        return

    import viser
    from viser.extras import ViserUrdf
    srv = viser.ViserServer(port=args.port)
    srv.scene.add_grid("/grid", width=2.0, height=2.0, cell_size=0.1)
    vr = ViserUrdf(srv, robot_urdf, root_node_name="/robot")
    srv.scene.add_frame("/robot", position=tuple(base), show_axes=False)

    # the fixed part rides a gizmo: the hand holding it, or the table it is on
    hold = srv.scene.add_transform_controls("/object", scale=0.25)
    handles = {}
    for i, (n, m) in enumerate(geoms.items()):
        handles[n] = srv.scene.add_mesh_simple(
            f"/object/{n}", vertices=rest[n] - centre,
            faces=np.asarray(m.triangles),
            color=tuple(int(255 * c) for c in PAL[i % len(PAL)]),
            flat_shading=False)
    tip = srv.scene.add_icosphere("/object/grasp", radius=0.012,
                                  color=(250, 240, 120))
    sl = srv.gui.add_slider("q", float(j["lower"]), float(j["upper"]),
                            float(span / 100), float(j["lower"]))
    play = srv.gui.add_button("sweep the joint")

    def draw(q):
        full = np.zeros(len(joints)); full[k] = q
        pose = link_poses(joints, root, full)
        for n, h in handles.items():
            T = pose.get(n, np.eye(4))
            h.vertices = (rest[n] - centre) @ T[:3, :3].T + T[:3, 3]
        T = pose.get(j["child"], np.eye(4))
        g = grasp_point(geoms, joints, j["child"])
        tip.position = tuple(T[:3, :3] @ g + T[:3, 3] - centre)
        i = int(np.clip(np.searchsorted(qs, q), 0, len(qs) - 1))
        vr.update_cfg(cfgs[i])

    sl.on_update(lambda _: draw(sl.value))
    hold.on_update(lambda _: None)      # the gizmo moves the whole object node

    @play.on_click
    def _(_):
        import time
        for q in np.concatenate([qs, qs[::-1]]):
            sl.value = float(q)
            time.sleep(0.06)

    draw(float(j["lower"]))
    print(f"[robot] serving on http://localhost:{args.port} -- drag the gizmo "
          f"to move the held part, or press 'sweep the joint'")
    import time
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()

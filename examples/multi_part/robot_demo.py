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


def grasp_point(geoms, joints, child, axis_frac=0.35):
    """Where the robot is holding: on the moving part, out along its axis.

    The centroid of a single-view shell sits behind the surface we actually
    saw, so the point is pushed out along the joint's own direction. It is a
    stand-in for the interaction point proper -- a handle, a lid edge -- which
    is step 6 and not yet recovered.
    """
    m = geoms.get(child)
    if m is None:
        return None
    v = np.asarray(m.vertices)
    c = v.mean(0)
    j = next((j for j in joints if j["child"] == child), None)
    if j is None:
        return c
    a = np.asarray(j["axis"], float)
    a = a / max(np.linalg.norm(a), 1e-9)
    return c + a * axis_frac * float(np.linalg.norm(v.max(0) - v.min(0)))


def solve_path(robot, targets, link_index, smooth=2.0, rest_w=0.01):
    """One configuration per target, smooth along the way.

    Each waypoint is an IK problem; the smoothness term between neighbours is
    what makes it a trajectory rather than a sequence of unrelated poses.
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
        if i:
            costs.append(pkc.smoothness_cost(v, vars_[i - 1], smooth))
    sol = (jaxls.LeastSquaresProblem(costs, vars_).analyze()
           .solve(verbose=False))
    return np.stack([np.asarray(sol[v]) for v in vars_])



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
    best = None
    for d in np.arange(0.30, 0.85, 0.05):
        b = np.array([-d, 0.0, -0.15])
        c = solve_path(robot, grip - b, ee_i)
        f = np.asarray(robot.forward_kinematics(c))[:, ee_i, 4:7] + b
        e = float(np.linalg.norm(f - grip, axis=1).max())
        if best is None or e < best[0]:
            best = (e, b, c)
    _, base, cfgs = best
    print(f"[robot] stood the base at {np.round(base, 2)} m")
    fk = np.asarray(robot.forward_kinematics(cfgs))[:, ee_i, 4:7] + base
    err = np.linalg.norm(fk - grip, axis=1)
    print(f"[robot] IK over {args.steps} waypoints: reach error median "
          f"{1000 * np.median(err):.1f} mm, worst {1000 * err.max():.1f} mm")
    step = np.abs(np.diff(cfgs, axis=0)).max(axis=1)
    print(f"[robot] largest joint step between waypoints: "
          f"{np.degrees(step.max()):.1f} deg")
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

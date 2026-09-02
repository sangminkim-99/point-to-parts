"""Reader for the RBO dataset of articulated objects.

RBO: Martin-Martin, Eppner and Brock, "The RBO Dataset of Articulated Objects
and Interactions", arXiv:1806.06465.

Mirrors the YCBInIsaacReader interface, but reads RBO's own on-disk layout
directly instead of a converted copy: poses come from the mocap CSV composed
through the tf chain, joint states from the object's joint-state CSV, and part
identity from the object's `<name>_in.yaml`.

Each **part** is exposed as an "object", so multi-part tracking and the existing
per-object evaluation line up without a separate format.

The one thing that is cached rather than read natively is depth: RBO stores it
as plain-text float matrices, 6.9 MB and 47 ms per frame versus 0.1 MB and 2 ms
for a uint16 PNG. `scripts/rbo/prepare_rbo_sequence.py` writes that cache into
`depth_png/` beside the original directory.
"""

import datetime
import functools
import glob
import logging
import os
import re

import cv2
import numpy as np
import pandas as pd
import trimesh
import yaml
from scipy.spatial.transform import Rotation

DEPTH_CACHE_DIR = "depth_png"

# /map -> camera_rgb_optical_frame, in composition order.
TF_CHAIN = [
    ("map", "AsusXtionCameraFrame"),
    ("AsusXtionCameraFrame", "camera_link"),
    ("camera_link", "camera_rgb_frame"),
    ("camera_rgb_frame", "camera_rgb_optical_frame"),
]


def depth2xyzmap(depth, K):
    invalid = depth < 0.1
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), sparse=False, indexing="ij")
    vs, us, zs = vs.reshape(-1), us.reshape(-1), depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    xyz = np.stack((xs, ys, zs), 1).reshape(H, W, 3).astype(np.float32)
    xyz[invalid] = 0
    return xyz


def _norm(frame):
    return str(frame).strip().lstrip("/")


def _pose_matrix(tx, ty, tz, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T[:3, 3] = (tx, ty, tz)
    return T


def stamp_from_name(path):
    """RBO names frames <index>-<unix_seconds>.<ext>; that stamp is the clock."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return float(stem.split("-", 1)[1])


def _nearest(sorted_times, t):
    j = int(np.searchsorted(sorted_times, t))
    cands = [k for k in (j - 1, j) if 0 <= k < len(sorted_times)]
    best = min(cands, key=lambda k: abs(sorted_times[k] - t))
    return best, abs(sorted_times[best] - t)


def _is_static(poses, trans_tol=1e-6, rot_tol_deg=1e-4):
    """True for calibration transforms that never move.

    RBO publishes the camera-internal links at ~10 Hz while images run at 30 Hz,
    so nearest-neighbour timing against them shows dt up to 100 ms even though
    they are bit-identical all sequence.  Gating on that silently discards half
    the frames, so static links must be detected and composed at their constant
    value instead.
    """
    if len(poses) < 2:
        return True
    if np.abs(poses[:, :3, 3] - poses[0, :3, 3]).max() > trans_tol:
        return False
    rel = np.einsum("ij,njk->nik", poses[0, :3, :3].T, poses[:, :3, :3])
    ang = [np.linalg.norm(Rotation.from_matrix(m).as_rotvec()) for m in rel]
    return np.degrees(ang).max() <= rot_tol_deg


def parse_origin(s):
    """'<origin xyz="..." rpy="..."/>' -> 4x4, URDF convention Rz*Ry*Rx."""
    xyz = re.search(r"xyz='([^']*)'", s) or re.search(r'xyz="([^"]*)"', s)
    rpy = re.search(r"rpy='([^']*)'", s) or re.search(r'rpy="([^"]*)"', s)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler(
        "xyz", [float(v) for v in rpy.group(1).split()] if rpy else np.zeros(3)
    ).as_matrix()
    T[:3, 3] = [float(v) for v in xyz.group(1).split()] if xyz else np.zeros(3)
    return T


def read_object_spec(models_dir, object_name, recorded_on=None):
    """Parse <object>_in.yaml: marker id -> part, joint graph, meshes.

    Most RBO objects ship several `configuration_<date>` directories, and they do
    NOT agree with each other: ikeasmall's three configs swap which marker id is
    the upper drawer and which is the lower.  The configuration must therefore be
    chosen by the sequence's recording date, not by taking the last one on disk.
    """
    root = os.path.join(models_dir, object_name)
    hits = sorted(
        os.path.join(d, f)
        for d, _, fs in os.walk(root)
        for f in fs
        if f == f"{object_name}_in.yaml"
    )
    if not hits:
        raise FileNotFoundError(f"{object_name}_in.yaml not found under {root}")

    chosen = hits[-1]
    if recorded_on is not None and len(hits) > 1:
        dated = []
        for h in hits:
            m = re.search(r"configuration_(\d{4}-\d{2}-\d{2})", h)
            if m:
                dated.append((datetime.date.fromisoformat(m.group(1)), h))
        if dated:
            dated.sort()
            earlier = [d for d in dated if d[0] <= recorded_on]
            # latest configuration at or before the recording, else the earliest
            chosen = (earlier[-1] if earlier else dated[0])[1]
            if not earlier:
                logging.warning(
                    "%s recorded %s predates every configuration %s; using %s",
                    object_name, recorded_on, [str(d[0]) for d in dated], chosen,
                )
    spec = yaml.safe_load(open(chosen))

    prefix = spec.get("link_name_prefix", "rb")
    id_to_part = {int(k): f"{prefix}{v}" for k, v in spec["link_ids"].items()}
    joints = []
    for j in spec.get("joints", []):
        parent, child = id_to_part[int(j["rb1_id"])], id_to_part[int(j["rb2_id"])]
        joints.append({
            "name": f"j_{parent[len(prefix):]}_{child[len(prefix):]}",
            "parent": parent,
            "child": child,
            "type": str(j["type"]).lower(),
        })

    # Order by link index, NOT marker id: link_ids reads e.g. {4:"0", 2:"1", 1:"2"},
    # so sorting on the key would put rb2 first and mislabel the base.
    parts = sorted(id_to_part.values(), key=lambda n: int(n[len(prefix):]))
    children = {j["child"] for j in joints}
    roots = [p for p in parts if p not in children]
    if len(roots) != 1:
        raise ValueError(f"expected one base part for {object_name}, got {roots}")

    return {
        "id_to_part": id_to_part,
        "parts": parts,
        "base_part": roots[0],
        "joints": joints,
        "meshes": spec.get("meshes", {}),
        "mesh_poses": spec.get("mesh_poses", {}),
        "base_marker": spec.get("base_marker"),
        "spec_path": chosen,
    }


class RBOReader:
    """RBO sequence directory, with each rigid part exposed as an object."""

    def __init__(
        self,
        video_dir,
        downscale=1,
        shorter_side=None,
        models_dir=None,
        max_sync_dt=0.05,
    ):
        self.video_dir = video_dir.rstrip("/")
        self.video_name = os.path.basename(self.video_dir)
        self.object_name = re.match(r"^([a-zA-Z]+)", self.video_name).group(1)
        self.models_dir = models_dir or os.path.join(
            os.path.dirname(os.path.dirname(self.video_dir)), "models"
        )
        self.max_sync_dt = max_sync_dt

        self.color_files = sorted(
            glob.glob(os.path.join(self.video_dir, "camera_rgb", "*.png"))
        )
        if not self.color_files:
            raise FileNotFoundError(f"no camera_rgb/*.png under {self.video_dir}")

        self.K, self.H, self.W, self.distortion = self._read_intrinsics()
        self.downscale = (
            shorter_side / min(self.H, self.W) if shorter_side is not None else downscale
        )
        self.H = int(self.H * self.downscale)
        self.W = int(self.W * self.downscale)
        self.K = self.K.copy()
        self.K[:2] *= self.downscale

        self.recorded_on = datetime.datetime.utcfromtimestamp(
            stamp_from_name(self.color_files[0])
        ).date()
        self.spec = read_object_spec(
            self.models_dir, self.object_name, recorded_on=self.recorded_on
        )
        self.object_names = list(self.spec["parts"])
        self.base_part = self.spec["base_part"]
        self.joints = self.spec["joints"]

        self._tf = self._read_tf()
        self._rb_times, self._rb_poses = self._read_rigid_bodies()
        self._js_times, self._js_states = self._read_joint_states()

        self._depth_files, self._depth_times, self._depth_is_png = self._find_depth()
        self.id_strs = [
            os.path.splitext(os.path.basename(f))[0] for f in self.color_files
        ]
        self._build_sync_index()

        self._mesh_cache = None

    # ---------- parsing ----------

    def _read_intrinsics(self):
        info = pd.read_csv(
            os.path.join(self.video_dir, "camera_rgb_camera_info.csv"), nrows=1
        )
        K = np.array([float(info[f"field.K{i}"].iloc[0]) for i in range(9)]).reshape(3, 3)
        return (
            K,
            int(info["field.height"].iloc[0]),
            int(info["field.width"].iloc[0]),
            [float(info[f"field.D{i}"].iloc[0]) for i in range(5)],
        )

    def _read_tf(self):
        tf = pd.read_csv(os.path.join(self.video_dir, "tf.csv"))
        p = "field.transforms0."
        parents = tf[p + "header.frame_id"].map(_norm).to_numpy()
        children = tf[p + "child_frame_id"].map(_norm).to_numpy()
        times = tf[p + "header.stamp"].to_numpy(dtype=np.float64) / 1e9
        cols = [p + "transform.translation." + a for a in "xyz"] + [
            p + "transform.rotation." + a for a in "xyzw"
        ]
        vals = tf[cols].to_numpy(dtype=np.float64)

        out = {}
        for parent, child in set(zip(parents, children)):
            sel = (parents == parent) & (children == child)
            order = np.argsort(times[sel])
            poses = np.stack([_pose_matrix(*v) for v in vals[sel][order]])
            out[(parent, child)] = (times[sel][order], poses, _is_static(poses))
        return out

    def _read_rigid_bodies(self):
        rb = pd.read_csv(os.path.join(self.video_dir, "rb_poses_array.csv"))
        slots = sorted(
            int(m.group(1))
            for m in (re.match(r"field\.markers(\d+)\.id$", c) for c in rb.columns)
            if m
        )
        times = rb["%time"].to_numpy(dtype=np.float64) / 1e9
        order = np.argsort(times)

        poses_by_part = {}
        for s in slots:
            marker_id = int(pd.Series(rb[f"field.markers{s}.id"]).mode().iloc[0])
            part = self.spec["id_to_part"].get(marker_id)
            if part is None:
                continue  # a tracked body that is not part of this object
            cols = [f"field.markers{s}.pose.position.{a}" for a in "xyz"] + [
                f"field.markers{s}.pose.orientation.{a}" for a in "xyzw"
            ]
            vals = rb[cols].to_numpy(dtype=np.float64)[order]
            poses_by_part[part] = np.stack([_pose_matrix(*v) for v in vals])
        return times[order], poses_by_part

    def _read_joint_states(self):
        path = os.path.join(self.video_dir, f"{self.object_name}_joint_states.csv")
        if not os.path.exists(path):
            return None, {}
        js = pd.read_csv(path)
        times = js["field.header.stamp"].to_numpy(dtype=np.float64) / 1e9
        order = np.argsort(times)
        states = {}
        for c in [c for c in js.columns if re.match(r"field\.name\d+$", c)]:
            k = int(c.rsplit("name", 1)[1])
            states[str(js[c].iloc[0])] = js[f"field.position{k}"].to_numpy(
                dtype=np.float64
            )[order]
        return times[order], states

    def _find_depth(self):
        cache = os.path.join(self.video_dir, DEPTH_CACHE_DIR)
        if os.path.isdir(cache):
            files = sorted(glob.glob(os.path.join(cache, "*.png")))
            if files:
                return files, np.array([stamp_from_name(f) for f in files]), True
            logging.warning("empty depth cache at %s", cache)
        files = sorted(
            glob.glob(os.path.join(self.video_dir, "camera_depth_registered", "*.txt"))
        )
        if not files:
            raise FileNotFoundError(f"no depth found under {self.video_dir}")
        logging.warning(
            "reading RBO text depth (~47 ms/frame); run "
            "scripts/rbo/prepare_rbo_sequence.py to cache it as PNG"
        )
        return files, np.array([stamp_from_name(f) for f in files]), False

    def _chain_pose(self, t):
        """T_map_camera_rgb_optical, plus the worst *dynamic* sync residual."""
        T, worst = np.eye(4), 0.0
        for link in TF_CHAIN:
            if link not in self._tf:
                raise KeyError(f"tf link {link} missing in {self.video_name}")
            times, poses, static = self._tf[link]
            if static:
                T = T @ poses[0]
                continue
            idx, dt = _nearest(times, t)
            T = T @ poses[idx]
            worst = max(worst, dt)
        return T, worst

    def _build_sync_index(self):
        """Resolve every RGB frame to its depth, mocap and joint-state sample."""
        self._depth_idx, self._rb_idx, self._js_idx = [], [], []
        self._T_cam_map, self.sync_dt = [], []
        for f in self.color_files:
            t = stamp_from_name(f)
            di, ddt = _nearest(self._depth_times, t)
            ri, rdt = _nearest(self._rb_times, t)
            T_map_cam, tfdt = self._chain_pose(t)
            self._depth_idx.append(di)
            self._rb_idx.append(ri)
            self._T_cam_map.append(np.linalg.inv(T_map_cam))
            self._js_idx.append(
                _nearest(self._js_times, t)[0] if self._js_times is not None else None
            )
            self.sync_dt.append(dict(depth=ddt, mocap=rdt, tf=tfdt))
        worst = max(max(d.values()) for d in self.sync_dt)
        if worst > self.max_sync_dt:
            logging.warning(
                "%s: worst sync residual %.3f s exceeds %.3f s",
                self.video_name, worst, self.max_sync_dt,
            )

    # ---------- interface parity with YCBInIsaacReader ----------

    @property
    def num_objects(self):
        return len(self.object_names)

    def get_video_name(self):
        return self.video_name

    def get_object_names(self):
        return list(self.object_names)

    def __len__(self):
        return len(self.color_files)

    def get_color(self, i):
        color = cv2.imread(self.color_files[i])[..., ::-1]
        return cv2.resize(color, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def get_depth(self, i):
        path = self._depth_files[self._depth_idx[i]]
        if self._depth_is_png:
            depth = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1e3
        else:
            depth = pd.read_csv(
                path, sep=r"\s+", header=None, dtype=np.float32
            ).to_numpy()
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return cv2.resize(depth, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def get_xyz_map(self, i):
        return depth2xyzmap(self.get_depth(i), self.K)

    def get_gt_pose(self, i, obj_name=None):
        obj_name = obj_name or self.object_names[0]
        poses = self._rb_poses.get(obj_name)
        if poses is None:
            logging.info("no mocap body for part %s", obj_name)
            return None
        return self._T_cam_map[i] @ poses[self._rb_idx[i]]

    def get_gt_poses(self, i):
        return {p: self.get_gt_pose(i, p) for p in self.object_names}

    def get_occ_mask(self, i):
        """RBO ships no hand annotation; occlusion must come from a segmenter."""
        return np.zeros((self.H, self.W), dtype=np.uint8)

    def get_mask(self, i, obj_name=None, use_init_mask=False):
        obj_name = obj_name or self.object_names[0]
        return self.render_masks(0 if use_init_mask else i)[obj_name]

    def get_masks(self, i, use_init_mask=False):
        rendered = self.render_masks(0 if use_init_mask else i)
        return [rendered[p] for p in self.object_names]

    def get_init_masks(self):
        return self.get_masks(i=0, use_init_mask=True)

    # ---------- articulation ----------

    def get_joint_states(self, i):
        if self._js_times is None:
            return {}
        k = self._js_idx[i]
        return {n: float(v[k]) for n, v in self._js_states.items()}

    def get_joint_state_trajectory(self):
        return {
            n: np.array([float(v[k]) for k in self._js_idx])
            for n, v in self._js_states.items()
        }

    # ---------- meshes and rendered masks ----------

    def get_gt_mesh(self, obj_name=None):
        """Part mesh in that part's own mocap frame."""
        obj_name = obj_name or self.object_names[0]
        return self._part_meshes()[obj_name].copy()

    def _part_meshes(self):
        if self._mesh_cache is None:
            self._mesh_cache = {}
            for part, rel in self.spec["meshes"].items():
                m = trimesh.load(os.path.join(self.models_dir, rel), force="mesh")
                # mesh_poses applies as given (mesh -> rigid-body frame); the
                # inverse reading leaves the geometry floating off the object.
                m.apply_transform(parse_origin(self.spec["mesh_poses"][part]))
                self._mesh_cache[part] = m
        return self._mesh_cache

    @functools.lru_cache(maxsize=8)
    def _render(self, i):
        """Raycast the part meshes at frame i; returns a per-pixel part index map.

        Inter-part occlusion is exact.  External occluders (hands) are NOT cut
        out, so these masks are amodal with respect to them.
        """
        import open3d as o3d

        meshes = self._part_meshes()
        parts = [p for p in self.object_names if p in meshes]
        scene = o3d.t.geometry.RaycastingScene()
        gid = {}
        for p in parts:
            T = self.get_gt_pose(i, p)
            if T is None:
                continue
            m = meshes[p].copy()
            m.apply_transform(T)
            gid[p] = scene.add_triangles(
                o3d.t.geometry.TriangleMesh(
                    o3d.core.Tensor(np.asarray(m.vertices), dtype=o3d.core.Dtype.Float32),
                    o3d.core.Tensor(np.asarray(m.faces), dtype=o3d.core.Dtype.UInt32),
                )
            )
        rays = scene.create_rays_pinhole(
            intrinsic_matrix=o3d.core.Tensor(self.K),
            extrinsic_matrix=o3d.core.Tensor(np.eye(4)),
            width_px=self.W,
            height_px=self.H,
        )
        return scene.cast_rays(rays)["geometry_ids"].numpy(), gid

    def render_masks(self, i):
        """{part: uint8 mask} for frame i, rendered from the GT meshes."""
        hit, gid = self._render(i)
        return {
            p: ((hit == gid[p]).astype(np.uint8) if p in gid
                else np.zeros((self.H, self.W), dtype=np.uint8))
            for p in self.object_names
        }

    def render_part_index_map(self, i):
        """int8 map: index into object_names, -1 where nothing is hit."""
        hit, gid = self._render(i)
        out = np.full((self.H, self.W), -1, dtype=np.int8)
        for k, p in enumerate(self.object_names):
            if p in gid:
                out[hit == gid[p]] = k
        return out

"""Reader for sequences rendered by scripts/sim/render_partnet_sequence.py.

Mirrors RBOReader's interface -- each part is exposed as an "object" -- so the
tracking pipeline and every evaluation script run on simulated and real data
without a branch.

Assets are PartNet-Mobility (sapien-sim/PartNetMobility), rendered with SAPIEN 3.
Unlike RBO, the ground truth here is exact: part poses come from the simulator,
part masks are the renderer's own segmentation buffer (modal, so occluders are
already excluded), and the depth noise is a known, settable quantity.
"""

import glob
import json
import os

import cv2
import numpy as np


def depth2xyzmap(depth, K):
    invalid = depth < 1e-6
    H, W = depth.shape[:2]
    vs, us = np.meshgrid(np.arange(H), np.arange(W), sparse=False, indexing="ij")
    vs, us, zs = vs.reshape(-1), us.reshape(-1), depth.reshape(-1)
    xs = (us - K[0, 2]) * zs / K[0, 0]
    ys = (vs - K[1, 2]) * zs / K[1, 1]
    xyz = np.stack((xs, ys, zs), 1).reshape(H, W, 3).astype(np.float32)
    xyz[invalid] = 0
    return xyz


class SapienReader:
    def __init__(self, video_dir, downscale=1, shorter_side=None, **kwargs):
        self.video_dir = video_dir.rstrip("/")
        self.video_name = os.path.basename(self.video_dir)
        self.meta = json.load(open(os.path.join(self.video_dir, "meta.json")))

        self.color_files = sorted(glob.glob(os.path.join(self.video_dir, "rgb", "*.png")))
        if not self.color_files:
            raise FileNotFoundError(f"no rgb/*.png under {self.video_dir}")
        self.id_strs = [os.path.splitext(os.path.basename(f))[0] for f in self.color_files]

        self.K = np.array(self.meta["intrinsics"], dtype=np.float64)
        H, W = self.meta["image_size"]
        self.downscale = (shorter_side / min(H, W)) if shorter_side else downscale
        self.H, self.W = int(H * self.downscale), int(W * self.downscale)
        self.K = self.K.copy()
        self.K[:2] *= self.downscale

        all_parts = list(self.meta["parts"])
        # PartNet-Mobility often carries a geometry-free "base" link. Counting it
        # as a ground-truth part would make every score look like it missed one.
        vis = np.zeros(len(all_parts), dtype=bool)
        probe = np.linspace(0, len(self.color_files) - 1, 5).astype(int)
        for i in probe:
            s_ = cv2.imread(os.path.join(self.video_dir, "seg",
                                         self.id_strs[i] + ".png"),
                            cv2.IMREAD_UNCHANGED)
            for k in range(len(all_parts)):
                if (s_ == k).sum() > 50:
                    vis[k] = True
        self._part_index = [k for k in range(len(all_parts)) if vis[k]]
        self.object_names = [all_parts[k] for k in self._part_index]
        self.hidden_parts = [all_parts[k] for k in range(len(all_parts)) if not vis[k]]
        if self.hidden_parts:
            print(f"[SapienReader] ignoring geometry-free parts: {self.hidden_parts}")
        self.joints = [
            {"name": j["name"],
             "type": "prismatic" if "prismatic" in j["type"] else "revolute",
             "parent": j["parent"], "child": j["child"], "limits": j["limits"]}
            for j in self.meta["joints"]
        ]
        self.joints = [j for j in self.joints
                       if j["child"] in self.object_names or j["parent"] in self.object_names]
        children = {j["child"] for j in self.joints}
        roots = [p for p in self.object_names if p not in children]
        self.base_part = roots[0] if roots else self.object_names[0]

        d = np.load(os.path.join(self.video_dir, "poses.npz"))
        self._T_cam_part = d["T_cam_part"]          # (T, K, 4, 4)
        self._q = d["joint_states"]                 # (T, J)
        self.cam_poses = d["cam_poses"]
        self.recorded_on = None
        # exposed so callers can report the conditions a number was measured under
        self.depth_noise = float(self.meta.get("depth_noise", 0.0))
        self.depth_quant_mm = float(self.meta.get("depth_quant_mm", 0.0))

    # ---------- interface parity ----------

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
        c = cv2.imread(self.color_files[i])[..., ::-1]
        return cv2.resize(c, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def _sibling(self, i, sub):
        return os.path.join(self.video_dir, sub, self.id_strs[i] + ".png")

    def get_depth(self, i):
        d = cv2.imread(self._sibling(i, "depth"), cv2.IMREAD_UNCHANGED)
        d = d.astype(np.float32) / 1000.0
        return cv2.resize(d, (self.W, self.H), interpolation=cv2.INTER_NEAREST)

    def get_xyz_map(self, i):
        return depth2xyzmap(self.get_depth(i), self.K)

    def get_gt_pose(self, i, obj_name=None):
        obj_name = obj_name or self.object_names[0]
        k = self._part_index[self.object_names.index(obj_name)]
        return self._T_cam_part[min(i, len(self._T_cam_part) - 1), k]

    def get_gt_poses(self, i):
        return {p: self.get_gt_pose(i, p) for p in self.object_names}

    def render_part_index_map(self, i):
        """int8 map: index into object_names, -1 for background."""
        s = cv2.imread(self._sibling(i, "seg"), cv2.IMREAD_UNCHANGED)
        s = cv2.resize(s, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        out = np.full(s.shape, -1, dtype=np.int8)
        for new_k, orig_k in enumerate(self._part_index):
            out[s == orig_k] = new_k
        return out

    def render_masks(self, i):
        pim = self.render_part_index_map(i)
        return {p: (pim == k).astype(np.uint8)
                for k, p in enumerate(self.object_names)}

    def get_mask(self, i, obj_name=None, use_init_mask=False):
        obj_name = obj_name or self.object_names[0]
        return self.render_masks(0 if use_init_mask else i)[obj_name]

    def get_masks(self, i, use_init_mask=False):
        m = self.render_masks(0 if use_init_mask else i)
        return [m[p] for p in self.object_names]

    def get_init_masks(self):
        return self.get_masks(i=0, use_init_mask=True)

    def get_occ_mask(self, i):
        return np.zeros((self.H, self.W), dtype=np.uint8)

    # ---------- articulation ----------

    def get_joint_states(self, i):
        k = min(i, len(self._q) - 1)
        return {j["name"]: float(self._q[k, n]) for n, j in enumerate(self.joints)}

    def get_joint_state_trajectory(self):
        return {j["name"]: self._q[:, n] for n, j in enumerate(self.joints)}

    # ---------- meshes ----------

    def _part_meshes(self):
        """Not used for simulated data: masks come from the renderer directly."""
        return {}

    def get_gt_mesh(self, obj_name=None):
        return None


def open_sequence(path, **kwargs):
    """Pick the right reader for a sequence directory."""
    if os.path.exists(os.path.join(path, "poses.npz")) and \
            os.path.exists(os.path.join(path, "meta.json")):
        return SapienReader(path, **kwargs)
    from point2pose.io.sources.dataset.rbo_reader import RBOReader
    return RBOReader(path, **kwargs)

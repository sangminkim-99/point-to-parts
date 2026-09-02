import numpy as np
import open3d as o3d
from typing import Dict


class Object:
    """
    Object class that keeps tracks of the 3D points belonging to the object.
    """

    def __init__(self, id: int):
        self.id = id

        # 3D points belonging to the object, represented in the first frame coordinate system
        self.key_points = np.empty((0, 3))  # Mx3
        self.kp_track_indices = np.empty(
            (0,), dtype=np.int64
        )  # M, global track IDs. this is object track index to global track ID

        # Dense inverse mapping: global track ID -> object row index (or -1 if missing)
        # we use np array so that we can use vectorized operations
        self.track_idx_2_obj_idx = np.full((0,), -1, dtype=np.int32)

        self.uncertainties = np.empty((0,))
        self.valid = np.empty((0,))  # M, bool
        self.num_keyframes = 0
        self.key_point_frames = np.empty(
            (0,), dtype=int
        )  # M, frame IDs when points were added

        self.keyframes = []
        # pose of the object
        # init_pose is the transformation from the object frame to the first frame
        # e.g. the estimated pose of the object in the first frame
        self.init_pose = np.eye(4)

        # pose is the transformation from the first frame to the current frame
        # T_i_0: transformation points from frame 0 -> frame i
        self.pose = np.eye(
            4
        )  # 4x4 transformation matrix from object frame to world frame
        self.lost = False
        # Consecutive frames this object has been flagged lost. Maintained by the
        # front end; used to decide when re-acquisition may relax the jump guard.
        self.lost_streak = 0

        self.init_bbox = None
        self.bbox = (
            None  # 3D bounding box of the object, represented in the object frame
        )

        # SDF / TSDF reconstruction state
        self.sdf_volume = None
        self.sdf_num_integrated = 0
        self.sdf = None

        self.omega = np.zeros(3)
        self.v = np.zeros(3)
        self.mean_residual = 0.0

    def add_key_points(
        self,
        new_key_points: np.ndarray,
        new_uncertainties: np.ndarray,
        new_valid: np.ndarray,
        new_indices: np.ndarray = None,
        frame_id: int = None,
    ):
        """
        Add new 3D points to the object.

        Args:
            new_key_points (np.ndarray): New 3D points to add, shape (M, 3).
            new_uncertainties (np.ndarray): Uncertainties associated with the new points, shape (M,).
            new_valid (np.ndarray): Valid mask for the new points, shape (M,).
            new_indices (np.ndarray): Global track IDs for the new points, shape (M,).
            frame_id (int): Frame ID when these points were added. If None, uses -1.
        """
        assert new_key_points.shape[1] == 3, "new_key_points should have shape (M, 3)"
        assert (
            new_key_points.shape[0] == new_uncertainties.shape[0]
        ), "new_key_points and new_uncertainties should have the same number of points"

        if frame_id is None:
            frame_id = -1

        if new_indices is None:
            raise ValueError(
                "[Object] new_indices is required when adding new key points"
            )

        old_m = self.key_points.shape[0]

        # append new key points to the object
        self.key_points = np.vstack((self.key_points, new_key_points))
        self.kp_track_indices = np.hstack((self.kp_track_indices, new_indices))
        self.uncertainties = np.hstack((self.uncertainties, new_uncertainties))
        self.valid = np.hstack((self.valid, new_valid))
        self.key_point_frames = np.hstack(
            (self.key_point_frames, np.full(new_key_points.shape[0], frame_id))
        )

        new_m = self.key_points.shape[0]
        # append new track_idx_2_obj_idx to the object

        max_tid = int(new_indices.max())
        # if the track_idx_2_obj_idx is not large enough, resize it
        if self.track_idx_2_obj_idx.size <= max_tid:
            old = self.track_idx_2_obj_idx
            new = np.full((max_tid + 1,), -1, dtype=old.dtype)
            new[: old.size] = old
            self.track_idx_2_obj_idx = new

        new_rows = np.arange(old_m, new_m, dtype=np.int32)
        self.track_idx_2_obj_idx[new_indices] = new_rows

    def rebuild_track_index(self):
        """Rebuild the global-track-id -> row lookup from kp_track_indices."""
        if self.kp_track_indices.size == 0:
            self.track_idx_2_obj_idx = np.full((0,), -1, dtype=np.int32)
            return
        n = int(self.kp_track_indices.max()) + 1
        self.track_idx_2_obj_idx = np.full((n,), -1, dtype=np.int32)
        self.track_idx_2_obj_idx[self.kp_track_indices] = np.arange(
            self.kp_track_indices.size, dtype=np.int32
        )

    def split_off(self, track_indices: np.ndarray, new_id: int):
        """Move the rows for `track_indices` into a new Object and return it.

        This is how a discovered part leaves its parent.  The child inherits the
        parent's current pose, because up to this instant the two were explained
        by the same rigid motion; they diverge from here on.  Keypoints keep
        their global track ids, so the tracker is untouched by the split.

        Returns None if none of the requested tracks belong to this object.
        """
        track_indices = np.asarray(track_indices, dtype=np.int64).reshape(-1)
        in_range = track_indices < self.track_idx_2_obj_idx.size
        tids = track_indices[in_range]
        if tids.size == 0:
            return None
        rows = self.track_idx_2_obj_idx[tids]
        rows = rows[rows >= 0]
        if rows.size == 0:
            return None

        child = Object(new_id)
        child.key_points = self.key_points[rows].copy()
        child.kp_track_indices = self.kp_track_indices[rows].copy()
        child.uncertainties = self.uncertainties[rows].copy()
        child.valid = self.valid[rows].copy()
        child.key_point_frames = self.key_point_frames[rows].copy()
        child.num_keyframes = self.num_keyframes
        # the child was part of the parent until now, so it inherits that history
        child.keyframes = list(self.keyframes)
        child.init_pose = self.init_pose.copy()
        child.pose = self.pose.copy()
        child.rebuild_track_index()

        keep = np.ones(self.key_points.shape[0], dtype=bool)
        keep[rows] = False
        self.key_points = self.key_points[keep]
        self.kp_track_indices = self.kp_track_indices[keep]
        self.uncertainties = self.uncertainties[keep]
        self.valid = self.valid[keep]
        self.key_point_frames = self.key_point_frames[keep]
        self.rebuild_track_index()
        return child

    def save_key_points_with_colors(self, save_path: str, current_frame_id: int = None):
        """
        Save key points as a colored point cloud with different colors for different frames.

        Args:
            save_path (str): Path to save the point cloud file (.ply format)
            current_frame_id (int): Current frame ID for coloring (if None, uses frame_id from key_point_frames)
        """

        if len(self.key_points) == 0:
            return

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.key_points)

        # Generate colors based on frame IDs
        if current_frame_id is None:
            frame_ids = self.key_point_frames
        else:
            frame_ids = self.key_point_frames.copy()
            # Update points added in current frame
            frame_ids[frame_ids == -1] = current_frame_id

        # Normalize frame IDs to [0, 1] for color mapping
        if len(np.unique(frame_ids)) > 1:
            min_frame = np.min(frame_ids)
            max_frame = np.max(frame_ids)
            normalized_frames = (frame_ids - min_frame) / (max_frame - min_frame)
        else:
            normalized_frames = np.ones_like(frame_ids) * 0.5

        # Create color map (using HSV to get distinct colors)
        colors = np.zeros((len(self.key_points), 3))
        for i, norm_frame in enumerate(normalized_frames):
            # Use HSV color space: hue varies with frame, saturation=1, value=1
            hue = norm_frame * 0.8  # Use 0.8 to avoid wrapping back to red
            # Convert HSV to RGB
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            colors[i] = [r, g, b]

        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save point cloud
        o3d.io.write_point_cloud(save_path, pcd)

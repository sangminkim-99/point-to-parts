"""Read a recording made by examples/realsense_tracking/record_rgbd.py.

    <dir>/rgb/000000.png     colour, BGR png
    <dir>/depth/000000.png   depth aligned to colour, uint16 millimetres
    <dir>/cam_K.txt          3x3 colour intrinsics
    <dir>/masks.npz          optional, written by annotate.py
"""
from pathlib import Path

import cv2
import numpy as np


class Recording:
    def __init__(self, path):
        self.root = Path(path).expanduser()
        self.rgb_files = sorted((self.root / "rgb").glob("*.png"))
        self.depth_files = sorted((self.root / "depth").glob("*.png"))
        if not self.rgb_files:
            raise FileNotFoundError(f"no rgb/*.png under {self.root}")
        self.K = np.loadtxt(self.root / "cam_K.txt")
        h, w = cv2.imread(str(self.rgb_files[0])).shape[:2]
        self.H, self.W = h, w
        self.masks = None
        npz = self.root / "masks.npz"
        if npz.exists():
            d = np.load(npz, allow_pickle=True)
            self.masks = {int(f): d["masks"][k]
                          for k, f in enumerate(d["frames"]) if k < len(d["masks"])}

    def __len__(self):
        return min(len(self.rgb_files), len(self.depth_files))

    def get_color(self, i):
        """RGB, as the rest of the pipeline expects."""
        return cv2.cvtColor(cv2.imread(str(self.rgb_files[i])), cv2.COLOR_BGR2RGB)

    def get_depth(self, i):
        """Metres."""
        d = cv2.imread(str(self.depth_files[i]), cv2.IMREAD_UNCHANGED)
        return d.astype(np.float32) / 1000.0

    def get_mask(self, i):
        return None if self.masks is None else self.masks.get(i)

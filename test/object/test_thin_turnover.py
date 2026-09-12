"""Thin-slab turnover control: profile math, output conventions, GT correctness.

The pure window_profile tests run anywhere. The render tests need sapien and a
working Vulkan device (the point2pose_model env), like test_sim_sequence.py;
they render tiny 16-20 frame sequences at 160x120 so the whole file stays fast.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.sim.thin_turnover_control import window_profile, parse_window

REPO = Path(__file__).resolve().parents[2]


# ---- pure profile math (no simulator) ----------------------------------------

def test_window_profile_flat_outside_and_exact_at_ends():
    q = window_profile(100, 2.0, 0.2, 0.6)
    assert np.all(q[:20] == 0.0)                     # flat before the window
    assert np.allclose(q[60:], 2.0)                  # flat at amount after it
    assert np.all(np.diff(q) >= -1e-12)              # monotone ease, no overshoot
    assert q.max() <= 2.0 + 1e-12


def test_window_profile_rejects_bad_windows():
    with pytest.raises(ValueError):
        window_profile(10, 1.0, 0.6, 0.2)            # reversed
    with pytest.raises(ValueError):
        window_profile(10, 1.0, -0.1, 0.5)           # out of range
    with pytest.raises(ValueError):
        window_profile(1, 1.0, 0.0, 1.0)             # too few frames
    assert parse_window("0.15,0.55") == (0.15, 0.55)


# ---- rendered sequences (sapien + Vulkan) -------------------------------------

sapien = pytest.importorskip("sapien")


def _render(tmp_path, variant, frames):
    out = tmp_path / variant
    cmd = [sys.executable, "-m", "scripts.sim.thin_turnover_control",
           "--out", str(out), "--variant", variant, "--frames", str(frames),
           "--width", "160", "--height", "120"]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return out


def _angle(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


@pytest.fixture(scope="module")
def rigid(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("thin"), "rigid", 16)


@pytest.fixture(scope="module")
def hinge(tmp_path_factory):
    return _render(tmp_path_factory.mktemp("thin"), "hinge", 20)


def test_rigid_output_matches_renderer_conventions(rigid):
    import cv2
    meta = json.loads((rigid / "meta.json").read_text())
    assert meta["parts"] == ["body"] and meta["joints"] == []
    assert meta["depth_storage_mm"] == 1.0
    assert meta["pixel_reprojection_max_px"] < 0.1
    for sub in ("rgb", "depth", "seg"):
        assert len(list((rigid / sub).glob("*.png"))) == meta["frames"]
    seg = cv2.imread(str(rigid / "seg" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert set(np.unique(seg)) <= {0, 255}           # part index + 255 background
    depth = cv2.imread(str(rigid / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16
    on = depth[seg == 0]
    assert on.size and 100 < np.median(on) < 5000    # object depth in plausible mm


def test_rigid_gt_rotation_matches_scripted_turnover(rigid):
    d = np.load(rigid / "poses.npz")
    T, C = d["T_cam_part"], d["cam_poses"]
    assert T.shape[1] == 1 and d["joint_states"].shape[1] == 0
    Tw0, Tw1 = C[0] @ T[0, 0], C[-1] @ T[-1, 0]
    total = _angle((Tw1 @ np.linalg.inv(Tw0))[:3, :3])
    meta = json.loads((rigid / "meta.json").read_text())
    assert abs(total - meta["turnover_deg"]) < 0.5   # GT equals the script


def test_rigid_passes_through_edge_on(rigid):
    d = np.load(rigid / "poses.npz")
    meta = json.loads((rigid / "meta.json").read_text())
    union = d["mask_px"].sum(axis=1)
    assert union.min() < 0.4 * union.max()           # pronounced edge-on collapse
    a, b = meta["turnover_window"]
    n = meta["frames"]
    assert a * (n - 1) <= meta["edge_on_frame"] <= b * (n - 1)   # inside the flip
    assert meta["union_px_min"] == int(union.min())  # meta agrees with poses.npz


def test_hinge_gt_relative_rotation_equals_joint_state(hinge):
    d = np.load(hinge / "poses.npz")
    T, Q = d["T_cam_part"], d["joint_states"]
    assert T.shape[1] == 2 and Q.shape[1] == 1
    rel0 = np.linalg.inv(T[0, 0]) @ T[0, 1]
    for t in range(len(T)):
        rel = np.linalg.inv(T[t, 0]) @ T[t, 1]
        drel = _angle((rel @ np.linalg.inv(rel0))[:3, :3])
        assert abs(drel - abs(np.degrees(Q[t, 0] - Q[0, 0]))) < 0.1
    assert np.degrees(np.ptp(Q)) > 100               # the cover really opened


def test_hinge_metadata_and_both_parts_visible_during_hinge(hinge):
    meta = json.loads((hinge / "meta.json").read_text())
    assert meta["parts"] == ["body", "cover"]
    (j,) = meta["joints"]
    assert j["type"] == "revolute" and j["parent"] == "body" and j["child"] == "cover"
    d = np.load(hinge / "poses.npz")
    ha, hb = meta["hinge_window"]
    n = meta["frames"]
    win = d["mask_px"][int(np.ceil(ha * (n - 1))):]
    # Both parts observable while the joint moves -- the property the azimuth
    # default was chosen for; a control where one part is invisible tests nothing.
    assert win[:, 0].min() > 0 and win[:, 1].min() > 0


def test_distinct_faces_break_appearance_symmetry_uniform_does_not(tmp_path):
    import cv2

    def mean_color(out, frame):
        rgb = cv2.imread(str(out / "rgb" / f"{frame:06d}.png"))
        seg = cv2.imread(str(out / "seg" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        return rgb[seg != 255].mean(axis=0)

    outs = {}
    for faces in ("uniform", "distinct"):
        out = tmp_path / faces
        cmd = [sys.executable, "-m", "scripts.sim.thin_turnover_control",
               "--out", str(out), "--variant", "rigid", "--frames", "12",
               "--width", "160", "--height", "120", "--faces", faces]
        r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        outs[faces] = float(np.abs(mean_color(out, 0) - mean_color(out, 11)).max())
        assert json.loads((out / "meta.json").read_text())["faces"] == faces
    assert outs["uniform"] < 8        # opposite face looks the same (symmetric)
    assert outs["distinct"] > 60      # opposite face clearly distinguishable


def test_union_mask_convention_feeds_tracker_input(rigid):
    import cv2
    meta = json.loads((rigid / "meta.json").read_text())
    seg = cv2.imread(str(rigid / "seg" / "000000.png"), cv2.IMREAD_UNCHANGED)
    union = (seg != 255)
    assert union.sum() == np.load(rigid / "poses.npz")["mask_px"][0].sum()
    assert "EVAL-ONLY" in meta["gt_note"]            # GT stays out of the tracker

from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from examples.multi_part.author_demo import ReferenceTracker
from examples.multi_part.author_state import (displayed_poses, save_snapshot, snapshot,
                                              union_mask, cloud_center_radius, frame_view)
from examples.multi_part.recording import Recording
from test.object.test_urdf_export import model, fk


def state():
    joint = model('revolute')
    root = np.eye(4)
    root[:3,:3] = Rotation.from_euler('xyz',[.3,.8,-.5]).as_matrix()
    root[:3,3] = [.4,-.1,.8]
    return SimpleNamespace(parts=[SimpleNamespace(part_id=2,parent=0,pose=root,observed=True,joint=None),
                                  SimpleNamespace(part_id=7,parent=0,pose=root @ joint.at(.2),observed=True,joint=joint)],
                           root=0,points=np.ones((2,3)),colors=np.ones((2,3)),labels=np.array([0,1]),K=np.eye(3),frame=12)


def test_reference_relative_motion_cancels_moving_body():
    s = state()
    actual = displayed_poses(s)
    np.testing.assert_allclose(actual[0],np.eye(4),atol=1e-12)
    np.testing.assert_allclose(actual[1],s.parts[1].joint.at(.2),atol=1e-12)
    movement = np.eye(4)
    movement[:3,:3] = Rotation.from_euler('z',1.1).as_matrix()
    movement[:3,3] = [-.2,.3,.1]
    for p in s.parts:
        p.pose = movement @ p.pose
    np.testing.assert_allclose(displayed_poses(s),actual,atol=1e-12)


def test_preview_does_not_change_tracker_poses_or_joint():
    s = state()
    old = np.stack([p.pose.copy() for p in s.parts])
    actual = displayed_poses(s,joint_index=1,fraction=1.)
    np.testing.assert_allclose(actual[1],s.parts[1].joint.at(.7),atol=1e-12)
    np.testing.assert_array_equal(np.stack([p.pose for p in s.parts]),old)


def test_reference_is_identity_not_minimum_motion():
    t = ReferenceTracker.__new__(ReferenceTracker)
    t.parts = [SimpleNamespace(part_id=5), SimpleNamespace(part_id=2)]
    t.reference_id = 2
    assert t._pick_root() == 1
    t.parts.reverse()
    assert t._pick_root() == 0


def test_mesh_free_export_preserves_joint_motion(tmp_path):
    s = state()
    result = save_snapshot(s,tmp_path/'version')
    xml = ET.parse(tmp_path/'version/object.urdf').getroot()
    assert not xml.findall('.//mesh') and not xml.findall('.//collision')
    assert result['urdf']
    for q in [-.4,.1,.7]:
        np.testing.assert_allclose(fk(xml,'part0',{'j0_1':q})['part1'],s.parts[1].joint.at(q),atol=2e-8)


def test_unfitted_part_saves_cloud_without_inventing_urdf(tmp_path):
    s = state()
    s.parts[1].joint = None
    result = save_snapshot(s,tmp_path/'version')
    assert result['urdf'] is None
    assert (tmp_path/'version/model.npz').exists()
    assert not (tmp_path/'version/object.urdf').exists()


def test_sparse_snapshot_is_independent():
    s = state()
    for j,p in enumerate(s.parts):
        p.idx = np.array([j])
    s.model = None
    s.anchor_xyz = np.ones((2,3))
    s.n = 3
    shot = snapshot(s)
    s.parts[0].pose[:] = 0
    assert shot.parts[0].pose[3,3] == 1
    np.testing.assert_array_equal(shot.labels,[0,1])


# ---- PNG-mask recording fallback ---------------------------------------------

def _write_recording(root, n=2, with_png_masks=True):
    """A minimal on-disk recording the online capture would produce."""
    for sub in ('rgb', 'depth'):
        (root / sub).mkdir(parents=True)
    if with_png_masks:
        (root / 'masks').mkdir()
    np.savetxt(root / 'cam_K.txt', np.eye(3))
    for i in range(n):
        name = f'{i:06d}.png'
        cv2.imwrite(str(root / 'rgb' / name), np.full((8, 10, 3), i + 1, np.uint8))
        cv2.imwrite(str(root / 'depth' / name), np.full((8, 10), 500, np.uint16))
        if with_png_masks:
            m = np.zeros((8, 10), np.uint8); m[2:5, 3:7] = 1 + i
            cv2.imwrite(str(root / 'masks' / name), m)


def test_png_mask_fallback_is_read_per_frame(tmp_path):
    _write_recording(tmp_path)
    r = Recording(tmp_path)
    m0, m1 = r.get_mask(0), r.get_mask(1)
    assert m0 is not None and m0.shape == (8, 10)
    assert int(m0.max()) == 1 and int(m1.max()) == 2      # per-frame, not shared
    assert bool((m0 > 0).any())


def test_missing_masks_return_none_not_crash(tmp_path):
    _write_recording(tmp_path, with_png_masks=False)
    r = Recording(tmp_path)
    assert r.get_mask(0) is None                           # neither npz nor png


def test_npz_masks_take_precedence_over_png(tmp_path):
    _write_recording(tmp_path)
    np.savez(tmp_path / 'masks.npz', frames=np.array([0]),
             masks=np.array([np.full((8, 10), 9, np.uint8)]))
    r = Recording(tmp_path)
    assert int(np.asarray(r.get_mask(0)).max()) == 9       # npz wins for frame 0
    # A frame absent from the npz still falls back to its per-frame PNG.
    assert int(r.get_mask(1).max()) == 2


# ---- union mask / framing helpers --------------------------------------------

def test_union_mask_matches_renderer_background_convention():
    seg = np.array([[255, 0], [1, 255]], np.uint8)          # 255 = background
    np.testing.assert_array_equal(union_mask(seg), [[0, 1], [1, 0]])
    assert union_mask(seg).sum() == 2                        # both parts, no bg


def test_frame_view_centres_on_cloud_and_clamps_distance():
    pts = np.random.RandomState(1).randn(500, 3) * 0.05 + np.array([0.1, 0., 0.55])
    c, radius = cloud_center_radius(pts)
    np.testing.assert_allclose(c, [0.1, 0., 0.55], atol=0.03)
    cam = frame_view(c, radius)
    np.testing.assert_allclose(cam['look_at'], c, atol=1e-9)   # aims at geometry
    d = np.linalg.norm(np.array(cam['position']) - c)
    assert 0.3 <= d <= 1.1                                     # clamped, non-degenerate
    assert frame_view(c, 0.0)['position'] != tuple(c)         # never coincident


# ---- reference-selection and lost-root honesty -------------------------------

def test_select_reference_rejects_a_vanished_identity():
    t = ReferenceTracker.__new__(ReferenceTracker)
    t.parts = [SimpleNamespace(part_id=5), SimpleNamespace(part_id=2)]
    with pytest.raises(ValueError):
        t.select_reference(9)                                 # not a live identity
    t.reference_id = 9
    with pytest.raises(RuntimeError):
        t._pick_root()                                        # gone -> explicit, no silent promote


def test_lost_root_is_saved_as_unobserved_not_tracked(tmp_path):
    s = state()
    s.parts[0].observed = False                               # reference pose lost / held
    meta = save_snapshot(s, tmp_path / 'v', source='replay')
    saved = json.loads((tmp_path / 'v' / 'metadata.json').read_text())
    assert saved['observed'][0] is False                     # not claimed as tracked
    assert saved['capture_source'] == 'replay'
    # A held pose is still a finite pose we can draw; it must not blow up.
    assert np.isfinite(displayed_poses(s)[0]).all()

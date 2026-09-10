import numpy as np
from scipy.spatial.transform import Rotation

from examples.multi_part.surface_memory import uncovered_surface, match_joint_surface


K = np.array([[240., 0, 120], [0, 240., 100], [0, 0, 1]])


def pose(angle):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", angle, degrees=True).as_matrix()
    T[:3, 3] = [0, 0, .7]
    return T


def panel_depth(T, thickness=0.):
    y, x = np.mgrid[:200, :240]
    rays = np.stack([(x - 120) / 240, (y - 100) / 240, np.ones_like(x)], -1)
    normal = T[:3, 2]
    z = (normal @ T[:3, 3] + thickness) / (rays @ normal)
    local = (rays * z[..., None] - T[:3, 3]) @ T[:3, :3]
    mask = (local[..., 0] >= .02) & (local[..., 0] <= .18) \
        & (np.abs(local[..., 1]) <= .12) & (z > 0)
    return np.where(mask, z, 0).astype(np.float32), mask.astype(np.uint8)


def points():
    x, y = np.meshgrid(np.linspace(.025, .175, 18), np.linspace(-.115, .115, 24))
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def test_new_face_is_seeded_even_with_many_tracks_on_base():
    mask = np.zeros((120, 200), np.uint8)
    mask[10:110, 10:190] = 1
    y, x = np.mgrid[15:110:10, 15:90:10]
    fresh = uncovered_surface(mask, mask.astype(float), np.c_[x.ravel(), y.ravel()])
    assert fresh[30:90, 120:180].all()
    assert not fresh[20:90, 20:75].any()


def test_opposite_face_without_any_image_correspondences():
    angles = np.arange(0, 181, 2)
    depth, mask = panel_depth(pose(140), thickness=.003)
    result = match_joint_surface(points(), [pose(a) for a in angles], depth, mask, K)
    assert result is not None
    assert abs(angles[result["index"]] - 140) <= 4


def test_partial_occlusion_does_not_drag_panel_to_occluder():
    depth, mask = panel_depth(pose(140))
    depth[:95] = .3   # foreground occluder, absent from object mask
    mask[:95] = 0
    angles = np.arange(0, 181, 2)
    result = match_joint_surface(points(), [pose(a) for a in angles], depth, mask, K)
    assert result is not None
    assert abs(angles[result["index"]] - 140) <= 4


def test_hidden_or_missing_surface_is_not_recovered():
    depth = np.full((200, 240), .3, np.float32)
    poses = [pose(a) for a in range(0, 181, 2)]
    assert match_joint_surface(points(), poses, depth, np.ones_like(depth), K) is None
    assert match_joint_surface(points(), poses, depth * 0, depth * 0, K) is None


def test_two_equally_supported_distinct_poses_are_ambiguous():
    a, b = pose(0), pose(180)
    da, ma = panel_depth(a)
    db, mb = panel_depth(b)
    result = match_joint_surface(points(), [a, b], np.maximum(da, db), ma | mb, K)
    assert result is None

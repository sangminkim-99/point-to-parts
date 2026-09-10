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


def recovery_tracker():
    from examples.multi_part.naive import NaivePart, NaiveConfig, NaivePartTracker
    from point2pose.pipeline.components.joint_model import JointModel
    cfg = NaiveConfig(surface_recovery=True)
    s = NaivePartTracker(K, cfg, None, None)
    s.model = None
    s.anchor_xyz = points()
    s.anchor_ok = np.ones(len(s.anchor_xyz), bool)
    parent = NaivePart(idx=np.array([], int), part_id=0)
    jm = JointModel()
    jm.kind = "revolute"
    jm.A0 = pose(0)
    jm._axis0 = np.array([0., 1., 0.])
    jm._point0 = np.zeros(3)
    jm.A = [pose(a) for a in range(0, 61, 5)]
    jm.conf = {"valid": True, "conf": .95}
    child = NaivePart(idx=np.arange(len(s.anchor_xyz)), part_id=7,
                     pose=pose(60), joint=jm, parent=0, observed=False)
    s.parts = [parent, child]
    return s


def test_recovery_keeps_identity_and_does_not_train_joint_on_its_predictions():
    s = recovery_tracker()
    child = s.parts[1]
    old_observations = np.stack(child.joint.A).copy()
    old_anchors = s.anchor_xyz.copy()
    depth, mask = panel_depth(pose(140), thickness=.003)
    s._recover_surfaces(depth, mask, 10)
    assert not child.surface_recovered  # temporal confirmation required
    s._recover_surfaces(depth, mask, 11)
    assert child.surface_recovered
    assert child.part_id == 7 and len(s.parts) == 2
    error = Rotation.from_matrix(child.pose[:3, :3].T @ pose(140)[:3, :3]).magnitude()
    assert np.degrees(error) < 4
    np.testing.assert_array_equal(np.stack(child.joint.A), old_observations)
    np.testing.assert_array_equal(s.anchor_xyz, old_anchors)


def test_stale_parent_cannot_confirm_child_recovery():
    s = recovery_tracker()
    s.parts[0].observed = False
    depth, mask = panel_depth(pose(140))
    for i in range(10, 14):
        s._recover_surfaces(depth, mask, i)
    assert not s.parts[1].surface_recovered


def test_missing_tracks_hold_pose_but_invalidate_measurement():
    s = recovery_tracker()
    p = s.parts[1]
    previous = p.pose.copy()
    p.observed, p.resid = True, .001
    s._fit(p, np.zeros_like(s.anchor_xyz), np.zeros(len(s.anchor_xyz), bool),
           np.zeros(len(s.anchor_xyz), bool))
    assert not p.observed and np.isinf(p.resid)
    np.testing.assert_array_equal(p.pose, previous)


def test_split_does_not_renumber_existing_part_identities():
    from examples.multi_part.naive import NaivePart, NaiveConfig, NaivePartTracker
    s = NaivePartTracker(K, NaiveConfig(), None, None)
    s.anchor_xyz = points()[:18]
    s.anchor_ok = np.ones(18, bool)
    s.track_born = np.zeros(18, int)
    s.co_same = np.zeros((18, 18))
    s.co_seen = np.zeros((18, 18))
    s.model = None
    s.parts = [NaivePart(idx=np.arange(12), part_id=10),
               NaivePart(idx=np.arange(12, 18), part_id=11)]
    s.next_part_id = 12
    s._reparent = lambda: None
    s._fit_box = lambda p: None
    s._accept(0, s.parts[0], [np.arange(6), np.arange(6, 12)],
              [np.eye(4), pose(20)], s.anchor_xyz, s.anchor_ok, "test", 1., None)
    assert [p.part_id for p in s.parts] == [10, 12, 11]


def test_trajectory_alignment_handles_new_frame_and_list_reordering():
    from examples.multi_part.evaluate import aligned_trajectory_errors
    class Reader:
        def get_gt_pose(self, frame, name):
            return pose(frame * 10)
    r = Reader()
    offset = pose(35)
    log = [(1, [pose(10) @ offset]),
           (2, [np.eye(4), pose(20) @ offset]),
           (3, [pose(30) @ offset, np.eye(4)])]
    ids = [[7], [9, 7], [7, 9]]
    result = aligned_trajectory_errors(r, [{"part": 0, "gt": "lid"}], log, ids, [7, 9])
    assert result[0]["frames"] == 3
    assert result[0]["translation_mm"] < 1e-8
    assert result[0]["rotation_deg"] < 1e-8
    # A fixed alignment must not erase real drift later in the sequence.
    for _, poses in log[1:]:
        for T in poses:
            T[0, 3] += .03
    result = aligned_trajectory_errors(r, [{"part": 0, "gt": "lid"}], log, ids, [7, 9])
    assert result[0]["translation_mm"] > 20


def test_retro_history_requires_support_from_whole_discovered_part():
    from types import SimpleNamespace
    from examples.multi_part.naive import NaivePart, NaiveConfig, NaivePartTracker
    reg = SimpleNamespace(_RANSAC=lambda **kwargs: {"T": np.eye(4)})
    s = NaivePartTracker(K, NaiveConfig(retro_min_frac=.6), None, reg)
    s.anchor_xyz = points()[:10]
    s.anchor_ok = np.ones(10, bool)
    s.frames = [(0, s.anchor_xyz[:3], np.ones(3, bool), np.ones(3, bool)),
                (1, s.anchor_xyz, np.ones(10, bool), np.ones(10, bool))]
    history = s._retro_hist(NaivePart(idx=np.arange(10)))
    assert [frame for frame, _, _ in history] == [1]


def test_reprojection_occlusion_is_not_motion_evidence():
    from examples.multi_part.surface_memory import reprojection_evidence
    K = np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1.]])
    depth = np.ones((40, 40), np.float32)
    mask = np.ones((40, 40), np.uint8)
    pts = np.array([[0., 0., 1.], [.03, 0, 1.], [0, .03, 1.]])
    T = np.eye(4)
    assert reprojection_evidence(pts, T, depth, mask, K)['support'] == 1
    T[2, 3] = .1
    hidden = reprojection_evidence(pts, T, depth, mask, K)
    assert hidden == {'support': 0., 'contradiction': 0., 'visible': 0.}
    T[2, 3] = -.1
    assert reprojection_evidence(pts, T, depth, mask, K)['contradiction'] == 1


def test_reprojection_invalid_depth_and_offscreen_are_neutral():
    from examples.multi_part.surface_memory import reprojection_evidence
    K = np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1.]])
    depth = np.zeros((40, 40), np.float32)
    mask = np.ones((40, 40), np.uint8)
    pts = np.array([[0., 0., 1.], [10, 10, 1.]])
    assert reprojection_evidence(pts, np.eye(4), depth, mask, K) == {
        'support': 0., 'contradiction': 0., 'visible': 0.}


def test_reprojection_silhouette_contradiction():
    from examples.multi_part.surface_memory import reprojection_evidence
    K = np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1.]])
    depth = np.ones((40, 40), np.float32)
    mask = np.zeros((40, 40), np.uint8)
    pts = np.array([[0., 0., 1.]])
    assert reprojection_evidence(pts, np.eye(4), depth, mask, K)['contradiction'] == 1


def test_symmetric_depth_ablation_counts_occlusion_as_error():
    from examples.multi_part.surface_memory import reprojection_evidence
    depth = np.ones((200, 240), np.float32)
    pts = np.array([[0., 0., 1.1]])
    result = reprojection_evidence(pts, np.eye(4), depth, np.ones_like(depth), K,
                                   occlusion_aware=False)
    assert result['contradiction'] == 1.


def _split_gate_scene(moving_depth):
    from types import SimpleNamespace
    from examples.multi_part.naive import NaiveConfig, NaivePartTracker
    tracker = NaivePartTracker.__new__(NaivePartTracker)
    tracker.cfg = NaiveConfig(split_reprojection_min_points=6)
    tracker.model = None
    x, y = np.meshgrid(np.linspace(-.03, .03, 3), np.linspace(-.03, .03, 3))
    base = np.column_stack([x.ravel() - .15, y.ravel(), np.ones(x.size)])
    moving = np.column_stack([x.ravel() + .15, y.ravel(), np.full(x.size, moving_depth)])
    tracker.anchor_xyz = np.concatenate([base, moving])
    tracker.K = K
    tracker.current_depth = np.ones((200, 240))
    tracker.mask = np.ones((200, 240), np.uint8)
    tracker.n = 10
    shifted = np.eye(4)
    shifted[2, 3] = 1 - moving_depth
    return tracker, SimpleNamespace(pose=np.eye(4)), [np.arange(9), np.arange(9, 18)], [np.eye(4), shifted]


def test_split_gate_requires_observable_contradiction_reduction():
    tracker, part, groups, motions = _split_gate_scene(.8)
    assert tracker._validate_split_reprojection(0, part, groups, motions)


def test_split_gate_does_not_use_hidden_geometry_as_articulation_evidence():
    tracker, part, groups, motions = _split_gate_scene(1.2)
    assert not tracker._validate_split_reprojection(0, part, groups, motions)

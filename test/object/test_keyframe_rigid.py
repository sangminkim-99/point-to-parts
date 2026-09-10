import numpy as np
from scipy.spatial.transform import Rotation
from examples.multi_part.keyframe_rigid import robust_rigid, fit_rigid, RGBDKeyframes, Keyframe


def test_ransac_recovers_se3_with_outliers_without_global_rng_changes():
    rng = np.random.default_rng(42)
    source = rng.uniform(-.2, .2, (100, 3))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler('xyz', [.15, -.2, .25]).as_matrix()
    T[:3, 3] = [.03, -.02, .01]
    target = source @ T[:3, :3].T + T[:3, 3]
    target[:35] = rng.uniform(-.4, .4, (35, 3))
    np.random.seed(18); expected = np.random.random(); np.random.seed(18)
    actual, inside, _ = robust_rigid(source, target)
    assert np.allclose(actual, T, atol=1e-8)
    assert inside.sum() == 65
    assert np.random.random() == expected


def test_collinear_and_insufficient_geometry_rejected():
    source = np.column_stack([np.arange(10), np.zeros((10, 2))])
    assert robust_rigid(source, source) is None
    assert robust_rigid(source[:3], source[:3]) is None


def test_planar_points_preserve_proper_rotation():
    source = np.array([[0., 0, 0], [.1, 0, 0], [0, .1, 0], [.1, .1, 0]])
    R = Rotation.from_euler('y', 145, degrees=True).as_matrix()
    T = fit_rigid(source, source @ R.T + [0, 0, .7])
    assert np.allclose(T[:3, :3], R)
    assert np.isclose(np.linalg.det(T[:3, :3]), 1)


def test_failed_frame_does_not_grow_or_overwrite_keyframe_memory():
    tracker = RGBDKeyframes(np.eye(3))
    tracker.previous = Keyframe(0, np.zeros((10, 3)), np.zeros((10, 128), np.float32), np.eye(4))
    tracker.keyframes = [tracker.previous]
    tracker.extract = lambda *args: (np.empty((0, 3)), np.empty((0, 128), np.float32))
    pose, observed, count, reference = tracker.step(1, None, None, None)
    assert not observed and count == 0 and reference == -1
    assert len(tracker.keyframes) == 1 and tracker.previous.frame == 0
    assert np.allclose(pose, np.eye(4))


def test_multiple_reference_frames_compose_into_original_camera_gauge():
    tracker = RGBDKeyframes(np.eye(3), multi_reference=True)
    points = np.random.default_rng(7).uniform(-.2, .2, (40, 3))
    transforms = []
    for angles, translation in [([25, 10, -15], [.1, .2, .4]),
                                ([-15, 30, 20], [-.2, .1, .5]),
                                ([40, -20, 35], [.3, -.1, .6])]:
        T = np.eye(4)
        T[:3, :3] = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
        T[:3, 3] = translation
        transforms.append(T)
    def moved(T):
        return points @ T[:3, :3].T + T[:3, 3]
    tracker.keyframes = [Keyframe(i, moved(T), None, T) for i, T in enumerate(transforms[:2])]
    tracker.previous = tracker.keyframes[-1]
    tracker.pose = transforms[1].copy()
    tracker.extract = lambda *args: (moved(transforms[2]), None)
    tracker.matches = lambda *args: np.column_stack([np.arange(40), np.arange(40)])
    pose, observed, _, _ = tracker.step(2, None, None, None)
    assert observed
    assert np.allclose(pose, transforms[2], atol=1e-8)

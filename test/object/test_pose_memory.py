import numpy as np
from examples.multi_part.pose_memory import choose_pose


def setup_scene():
    K = np.array([[100., 0, 20], [0, 100., 20], [0, 0, 1.]])
    x, y = np.meshgrid(np.linspace(-.1, .1, 10), np.linspace(-.1, .1, 10))
    points = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
    return points, np.ones((40, 40)), np.ones((40, 40)), K


def test_supported_prediction_rescues_free_space_fit():
    points, depth, mask, K = setup_scene()
    sparse = np.eye(4)
    sparse[2, 3] = -.1
    T, rescued = choose_pose(points, sparse, np.eye(4), depth, mask, K, lambda T: T)
    assert rescued and np.allclose(T, np.eye(4))


def test_fully_hidden_memory_does_not_override_sparse():
    points, depth, mask, K = setup_scene()
    sparse, prediction = np.eye(4), np.eye(4)
    sparse[2, 3], prediction[2, 3] = .1, .2
    T, rescued = choose_pose(points, sparse, prediction, depth, mask, K, lambda T: T)
    assert not rescued and np.allclose(T, sparse)


def test_supported_sparse_pose_skips_optimizer():
    points, depth, mask, K = setup_scene()
    def forbidden(T):
        raise AssertionError('unnecessary refinement')
    T, rescued = choose_pose(points, np.eye(4), np.eye(4), depth, mask, K, forbidden)
    assert not rescued


def test_continuity_breaks_depth_ambiguity_with_supported_prediction():
    from scipy.spatial.transform import Rotation
    points, depth, mask, K = setup_scene()
    sparse = np.eye(4)
    sparse[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    T, rescued = choose_pose(points, sparse, np.eye(4), depth, mask, K, lambda T: T,
                              previous_pose=np.eye(4))
    assert rescued and np.allclose(T, np.eye(4))


def test_continuity_never_rescues_an_unsupported_prediction():
    from scipy.spatial.transform import Rotation
    points, depth, mask, K = setup_scene()
    sparse = np.eye(4)
    sparse[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    hidden = np.eye(4)
    hidden[2, 3] = .2
    T, rescued = choose_pose(points, sparse, hidden, depth, mask, K, lambda T: T,
                              previous_pose=np.eye(4))
    assert not rescued and np.allclose(T, sparse)


def test_incremental_composition_and_rng_isolation():
    from examples.multi_part.pose_memory import incremental_candidate
    from scipy.spatial.transform import Rotation
    points, _, _, _ = setup_scene()
    delta = np.eye(4)
    delta[:3, :3] = Rotation.from_euler('y', 3, degrees=True).as_matrix()
    delta[:3, 3] = [.01, 0, 0]
    current = points @ delta[:3, :3].T + delta[:3, 3]
    old_pose = np.eye(4)
    old_pose[:3, 3] = [0, .1, 0]
    class Register:
        def _RANSAC(self, **kw):
            np.random.random(30)
            return {'T': delta}
    np.random.seed(14)
    expected = np.random.random()
    np.random.seed(14)
    result = incremental_candidate(points, current, old_pose, Register())
    assert np.allclose(result, delta @ old_pose)
    assert np.random.random() == expected
    assert incremental_candidate(points, current + .1, old_pose, Register()) is None
    assert incremental_candidate(points, current, old_pose, Register(), max_step_deg=1) is None

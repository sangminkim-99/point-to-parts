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

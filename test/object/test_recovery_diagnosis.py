import numpy as np
from scripts.sim.diagnose_recovery import support_labels


def test_support_labels_count_visible_depth_matches_only():
    depth = np.ones((3, 3)); labels = np.full((3, 3), -1)
    labels[0, 0] = 0; labels[0, 1] = 1
    points = np.array([[0., 0, 1], [1., 0, 1], [0, 0, 1.2], [20, 0, 1], [2, 2, 1]])
    assert support_labels(points, np.eye(4), depth, np.eye(3), labels, .012) == {0: 1, 1: 1}


def test_estimated_ownership_requires_current_depth_agreement():
    from examples.multi_part.surface_memory import unowned_depth_mask
    depth = np.ones((20, 20)); mask = np.ones((20, 20))
    K = np.array([[10., 0, 10], [0, 10., 10], [0, 0, 1]])
    points = np.array([[0., 0, 1.]])
    actual = unowned_depth_mask(mask, depth, K, [(points, np.eye(4))])
    assert not actual[10, 10] and actual[0, 0]
    # Hidden or free-space geometry cannot claim this depth surface.
    for z in [.8, 1.2]:
        actual = unowned_depth_mask(mask, depth, K, [(points*np.array([1, 1, z]), np.eye(4))])
        assert actual.all()
    assert unowned_depth_mask(mask, depth, K, []).all()

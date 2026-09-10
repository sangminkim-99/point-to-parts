import numpy as np
from examples.multi_part.pose_memory import joint_pose_supported


def test_joint_override_needs_depth_support_and_must_not_worsen_free_pose():
    points = np.zeros((50, 3)); points[:, 2] = 1.
    depth = np.ones((3, 3)); mask = np.ones((3, 3))
    good = np.eye(4)
    bad = np.eye(4); bad[2, 3] = -.2
    hidden = np.eye(4); hidden[2, 3] = .2
    assert joint_pose_supported(points, bad, good, depth, mask, np.eye(3))
    assert not joint_pose_supported(points, good, bad, depth, mask, np.eye(3))
    assert not joint_pose_supported(points, bad, hidden, depth, mask, np.eye(3))
    assert not joint_pose_supported(points[:3], bad, good, depth, mask, np.eye(3))

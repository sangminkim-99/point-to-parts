from types import SimpleNamespace as NS
import numpy as np
from scipy.spatial.transform import Rotation
from examples.multi_part.naive import NaiveConfig, NaivePartTracker


def test_relock_jump_holds_pose_and_does_not_append_joint_history():
    tracker=NaivePartTracker.__new__(NaivePartTracker)
    tracker.cfg=NaiveConfig(pose_relock_max_deg=20.)
    tracker.anchor_xyz=np.array([[0,0,1.],[.1,0,1.],[0,.1,1.],[.1,.1,1.]])
    tracker.anchor_ok=np.ones(4,bool)
    candidate=np.eye(4); candidate[:3,:3]=Rotation.from_euler('y',100,degrees=True).as_matrix()
    tracker.reg=NS(_RANSAC=lambda **kw: {'T':candidate})
    tracker.n=48
    part=NS(observed=False,idx=np.arange(4),pose=np.eye(4),part_id=7,
            over=3,hist=[('sentinel',)],t_lo=None,t_hi=None)
    cur=tracker.anchor_xyz @ candidate[:3,:3].T
    for _ in range(2):
        tracker._fit(part,cur,np.ones(4,bool),np.ones(4,bool))
        assert not part.observed and np.isinf(part.resid)
        assert part.over == 0 and part.hist == [('sentinel',)]
        np.testing.assert_array_equal(part.pose,np.eye(4))
    assert tracker.relock_rejections == 2


def test_relock_guard_default_is_disabled():
    assert NaiveConfig().pose_relock_max_deg == 0.

import numpy as np
from examples.multi_part.keyframe_rigid import RGBDKeyframes


def test_keyframe_lift_rejects_outside_and_nonfinite_features():
    tracker=RGBDKeyframes.__new__(RGBDKeyframes)
    tracker.K=np.eye(3)
    xy=np.array([[1.,1.],[-.1,1.],[3.,1.],[np.nan,1.],[np.inf,1.],[2.9,1.]])
    points,good=tracker.lift(xy,np.ones((3,3)),np.ones((3,3)))
    np.testing.assert_array_equal(good,[True,False,False,False,False,False])
    np.testing.assert_array_equal(points,[[1.,1.,1.]])


def test_keyframe_lift_empty_features():
    tracker=RGBDKeyframes.__new__(RGBDKeyframes);tracker.K=np.eye(3)
    points,good=tracker.lift(np.empty((0,2)),np.ones((3,3)),np.ones((3,3)))
    assert points.shape == (0,3) and good.shape == (0,)

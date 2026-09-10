"""Regression checks for scripted simulation GT, without a renderer/GPU."""
import numpy as np
import pytest
from scripts.sim.render_partnet_sequence import joint_trajectory, motion_groups


def test_open_close_reaches_both_limits_and_holds():
    q = joint_trajectory(120, -.7, 1.2, mode="open_close")
    assert np.all(q[:24] == -.7) and np.all(q[-24:] == -.7)
    assert np.isclose(q.max(), 1.2)
    assert np.allclose(q, q[::-1])
    assert np.all(np.diff(q[:60]) >= 0)
    assert np.all(np.diff(q[60:]) <= 0)


def test_static_and_inactive_links_merge():
    names = ["root", "base", "lid", "handle", "drawer"]
    joints = [("fixed", "root", "base"), ("hinge", "base", "lid"),
              ("handle_fixed", "lid", "handle"), ("slide", "base", "drawer")]
    assert motion_groups(names, joints, set()) == {"root": names}
    assert motion_groups(names, joints, {"slide"}) == {
        "root": ["root", "base", "lid", "handle"], "drawer": ["drawer"]}
    assert motion_groups(names, joints, {"hinge", "slide"}) == {
        "root": ["root", "base"], "lid": ["lid", "handle"], "drawer": ["drawer"]}
    assert np.all(joint_trajectory(20, .3, 1.1, mode="static") == .3)


def test_grouping_independent_of_joint_order():
    joints = [("a", "root", "mid"), ("b", "mid", "end")]
    assert motion_groups(["root", "mid", "end"], joints[::-1], set()) == {
        "root": ["root", "mid", "end"]}


@pytest.mark.parametrize("kwargs", [{"n": 2}, {"static_prefix": .5}, {"open_frac": 0}])
def test_invalid_trajectory(kwargs):
    args = dict(n=120, lo=0, hi=1, mode="open_close")
    args.update(kwargs)
    with pytest.raises(ValueError):
        joint_trajectory(**args)

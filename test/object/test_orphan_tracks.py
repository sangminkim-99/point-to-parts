from types import SimpleNamespace as NS
import numpy as np
from examples.multi_part.orphan_tracks import OrphanTracks


def parts():
    return [NS(part_id=i, idx=np.array([],int), observed=True, sigma=.004, pose=np.eye(4)) for i in (4,8)]


def run(moving=True, hidden=False):
    a=OrphanTracks(); p=parts(); out=[]
    for f in range(6):
        p[1].pose[0,3]=f*.01 if moving else 0
        p[0].observed=not hidden
        out=a.update(f,np.array([[p[1].pose[0,3],0,1.]]),np.array([True]),np.array([True]),p)
    return out


def test_reobserved_track_follows_existing_moving_part():
    out=run(); assert len(out)==1 and out[0][1]==1
    np.testing.assert_allclose(out[0][2],[0,0,1],atol=1e-8)


def test_common_motion_is_ambiguous():
    assert not run(moving=False)


def test_held_competitor_does_not_lose_by_default():
    assert not run(hidden=True)


def test_invisible_and_pending_tracks_are_not_adopted():
    a=OrphanTracks();p=parts()
    for f in range(12):
        p[1].pose[0,3]=f*.01
        assert not a.update(f,np.array([[f*.01,0,1.]]),np.array([False]),np.array([True]),p)
        assert not a.update(f,np.array([[f*.01,0,1.]]),np.array([True]),np.array([True]),p,excluded=[0])


def test_repeated_frame_is_not_multiple_observations():
    a=OrphanTracks();p=parts()
    for _ in range(10):
        assert not a.update(0,np.array([[0,0,1.]]),np.array([True]),np.array([True]),p)
    assert len(a.history[0])==1

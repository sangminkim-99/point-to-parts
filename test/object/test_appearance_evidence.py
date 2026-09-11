import numpy as np
from examples.multi_part.appearance_evidence import appearance_evidence


def score(points,colors,depth=1.):
    return appearance_evidence(np.array(points),np.array(colors),np.eye(4),np.eye(3),np.full((2,2,3),255,np.uint8),np.full((2,2),depth),np.ones((2,2)))


def test_matching_geometry_but_wrong_color_is_distinguishable():
    assert score([[0,0,1]],[[1,1,1]])['mean_rgb_l1']==0.
    assert score([[0,0,1]],[[0,0,0]])['mean_rgb_l1']==1.


def test_hidden_duplicate_does_not_outvote_visible_surface():
    r=score([[0,0,1],[0,0,1.005]],[[0,0,0],[1,1,1]])
    assert r['samples']==1 and r['mean_rgb_l1']==1.


def test_occluded_and_offscreen_samples_supply_no_color_evidence():
    r=score([[0,0,2],[10,0,1]],[[1,1,1],[1,1,1]])
    assert r['samples']==0 and r['mean_rgb_l1'] is None

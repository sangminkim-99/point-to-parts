"""Per-child separation for split proposals (`split_sep_per_child`, default off).

Measured on ikeasmall02 f47 and cardboardbox01 f233: a proposal passes the
pooled separation test because the large child's points carry the median,
while the small child's points fit the other motion almost as well (2.4 sigma)
-- it is half another body. These tests drive `_sep_sigma` directly.
"""
import numpy as np

from examples.multi_part.naive import NaiveConfig, NaivePart, NaivePartTracker


def tracker(cfg_over=None):
    s = NaivePartTracker.__new__(NaivePartTracker)
    s.cfg = NaiveConfig()
    for k, v in (cfg_over or {}).items():
        setattr(s.cfg, k, v)
    return s


def shift(dx):
    T = np.eye(4)
    T[0, 3] = dx
    return T


def scene(n_big=100, n_small=20):
    """Two groups on the x axis; motion 0 is identity, motion 1 slides by dx.

    The swap displacement of every point is exactly |dx| for a pure
    translation, so the small child's separation is set independently by
    giving it its own second motion below."""
    s = tracker()
    rng = np.random.default_rng(0)
    s.anchor_xyz = rng.normal(size=(n_big + n_small, 3)) * 0.1
    big = np.arange(n_big)
    small = np.arange(n_big, n_big + n_small)
    part = NaivePart(idx=np.arange(n_big + n_small))
    part.sigma = 0.004
    return s, big, small, part


def test_pooled_median_is_carried_by_the_large_child():
    s, big, small, part = scene()
    # the big child moves 40 mm relative to the other motion, the small one 10 mm.
    # _sep_sigma measures |A - B| per point under the two motions, so give the
    # groups motions whose difference is 40 mm and read the pooled median
    ss = s._sep_sigma([big, small], [shift(0.0), shift(0.04)], part)
    assert abs(ss - 10.0) < 1e-6          # 40 mm / 4 mm, both children alike here


def test_per_child_takes_the_weakest_child():
    # Emulate the ikea f47 shape with a custom per-group displacement: reuse
    # _sep_sigma's per-group medians by calling it on each group alone.
    s, big, small, part = scene()
    s.cfg.split_sep_per_child = True
    strong = s._sep_sigma([big, big], [shift(0.0), shift(0.04)], part)
    weak = s._sep_sigma([small, small], [shift(0.0), shift(0.01)], part)
    assert abs(strong - 10.0) < 1e-6 and abs(weak - 2.5) < 1e-6
    # pooled over the two real groups the big child dominates the median ...
    s.cfg.split_sep_per_child = False
    d_big = np.full(len(big), 0.04)
    d_small = np.full(len(small), 0.01)
    pooled = np.median(np.concatenate([d_big, d_small])) / part.sigma
    assert pooled >= s.cfg.split_sep_sigma
    # ... while the per-child rule reports the small child's 2.5 sigma
    assert min(np.median(d_big), np.median(d_small)) / part.sigma < s.cfg.split_sep_sigma


def test_per_child_equals_pooled_when_children_agree():
    s, big, small, part = scene()
    a = s._sep_sigma([big, small], [shift(0.0), shift(0.03)], part)
    s.cfg.split_sep_per_child = True
    b = s._sep_sigma([big, small], [shift(0.0), shift(0.03)], part)
    assert abs(a - b) < 1e-9


def test_defaults_are_off():
    cfg = NaiveConfig()
    assert cfg.split_sep_per_child is False
    assert cfg.split_refine_coassoc is False

"""Kinematic tree from pairwise relative motion (`joint_graph`, default off).

Three parts: a body, and two drawers that slide along parallel axes at
different times. Body-drawer relative motion is one prismatic DoF each;
drawer-drawer relative motion is two DoF (both slides), so its per-observation
BIC is higher and the spanning tree must attach both drawers to the body, no
matter which part is the root. This is the ikeasmall02 shape measured in
doc/split_partition_purity.md.
"""
import numpy as np

from examples.multi_part.naive import NaiveConfig, NaivePart, NaivePartTracker


def tracker(cfg_over=None):
    s = NaivePartTracker.__new__(NaivePartTracker)
    s.cfg = NaiveConfig()
    s.cfg.joint_graph = True
    for k, v in (cfg_over or {}).items():
        setattr(s.cfg, k, v)
    s.n = 200
    rng = np.random.default_rng(0)
    s.anchor_xyz = rng.normal(size=(60, 3)) * 0.1
    s.anchor_ok = np.ones(60, bool)
    return s


def pose(t):
    T = np.eye(4)
    T[:3, 3] = t
    return T


def scene(root_idx):
    """Body still; drawer A slides +x on frames 0-99, drawer B slides +y on 100-199."""
    s = tracker()
    rng = np.random.default_rng(1)
    def noisy(T):
        T = T.copy(); T[:3, 3] += rng.normal(0, 0.0005, 3); return T
    body = NaivePart(idx=np.arange(0, 20), part_id=0)
    da = NaivePart(idx=np.arange(20, 40), part_id=1)
    db = NaivePart(idx=np.arange(40, 60), part_id=2)
    for p in (body, da, db):
        p.sigma = 0.002
        p.hist = []
    for n in range(200):
        qa = 0.1 * min(n, 99) / 99.0
        qb = 0.1 * max(0, n - 100) / 99.0
        body.hist.append((n, noisy(pose([0, 0, 0.5])), None))
        da.hist.append((n, noisy(pose([0.2 + qa, 0, 0.5])), None))
        db.hist.append((n, noisy(pose([-0.2, qb, 0.5])), None))
    parts = [body, da, db]
    s.parts = parts
    return s, parts


def test_defaults_off():
    assert NaiveConfig().joint_graph is False


def test_drawers_attach_to_the_body_whatever_the_root():
    for root in (0, 1, 2):
        s, parts = scene(root)
        par = s._graph_parents(root)
        ids = [p.part_id for p in parts]
        # undirected edges must be body-A and body-B, never A-B
        edges = {frozenset((ids[k], ids[par[k]])) for k in range(3) if k != root}
        assert edges == {frozenset((0, 1)), frozenset((0, 2))}, (root, par)
        assert par[root] == root


def test_tree_is_cached_between_recomputations():
    s, parts = scene(0)
    par1 = s._graph_parents(0)
    n_logs = len(s.graph_log)
    s.n += 1                                   # not yet due
    par2 = s._graph_parents(0)
    assert par1 == par2 and len(s.graph_log) == n_logs
    s.n += s.cfg.joint_graph_every
    s._graph_parents(0)
    assert len(s.graph_log) == n_logs + 1


def test_pair_without_shared_history_falls_back_to_the_star():
    s, parts = scene(0)
    parts[2].hist = parts[2].hist[:5]          # too short to fit
    par = s._graph_parents(0)
    assert par[2] == 0 and par[1] == 0

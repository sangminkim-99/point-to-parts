from types import SimpleNamespace as NS
import numpy as np
import torch
from examples.multi_part.split_evidence import SplitEvidence


def test_snapshot_is_causal_bounded_and_survives_live_map_changes():
    p = NS(part_id=7, observed=True, over=0)
    model = NS(labels=np.zeros(20, int), cloud=NS(means=torch.arange(60).reshape(20, 3).float()))
    memory = SplitEvidence(max_points=6)
    memory.update([p], model, 10)
    assert not memory.snapshots
    p.over = 1
    memory.update([p], model, 11)
    frame, points = memory.snapshots[7]
    expected = points.copy()
    assert frame == 11 and len(points) <= 6
    model.cloud.means += 100
    model.labels[:] = -1
    p.over = 0
    memory.update([p], model, 12)
    np.testing.assert_array_equal(memory.snapshots[7][1], expected)


def test_topology_change_invalidates_old_ownership_and_held_pose_cannot_capture():
    p = NS(part_id=7, observed=True, over=1)
    model = NS(labels=np.zeros(4, int), cloud=NS(means=torch.ones(4, 3)))
    memory = SplitEvidence()
    memory.update([p], model, 10)
    assert 7 in memory.snapshots
    p.observed = False
    child = NS(part_id=8, observed=False, over=1)
    memory.update([p, child], model, 11)
    assert not memory.snapshots


def test_gate_scores_retained_geometry_after_live_geometry_is_replaced():
    from test.object.test_surface_memory import _split_gate_scene
    tracker, part, groups, motions = _split_gate_scene(.8)
    part.part_id = 7
    tracker.cfg.split_reprojection_dense = True
    tracker.cfg.split_reprojection_frozen = True
    retained = tracker.anchor_xyz.copy()
    tracker._split_evidence = SplitEvidence()
    tracker._split_evidence.snapshots[7] = (1, retained)
    # Live maintenance has replaced the old face by the currently observed
    # surface. The old face must still provide the gate's contradiction.
    live = retained.copy()
    live[:, 2] = 1.
    tracker.model = NS(labels=np.zeros(len(live), int),
                       cloud=NS(means=torch.tensor(live)))
    assert tracker._validate_split_reprojection(0, part, groups, motions)
    tracker.cfg.split_reprojection_frozen = False
    assert not tracker._validate_split_reprojection(0, part, groups, motions)


def test_unsupported_snapshot_can_retry_original_live_gate():
    from test.object.test_surface_memory import _split_gate_scene
    tracker, part, groups, motions = _split_gate_scene(.8)
    part.part_id = 7
    tracker.cfg.split_reprojection_dense = True
    tracker.cfg.split_reprojection_frozen = True
    tracker.model = NS(labels=np.zeros(18,int),cloud=NS(means=torch.tensor(tracker.anchor_xyz)))
    stale=tracker.anchor_xyz.copy();stale[:,2]=2.
    tracker._split_evidence=SplitEvidence()
    tracker._split_evidence.snapshots[7]=(1,stale)
    assert not tracker._validate_split_reprojection(0,part,groups,motions)
    tracker.cfg.split_reprojection_frozen_fallback=True
    assert tracker._validate_split_reprojection(0,part,groups,motions)
    # Fallback still requires live evidence: it cannot force an acceptance.
    tracker.model.cloud.means[:,2]=2.
    assert not tracker._validate_split_reprojection(0,part,groups,motions)

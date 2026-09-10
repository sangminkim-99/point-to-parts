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

"""Ownership bookkeeping when a part splits off its parent object.

A split moves keypoint rows between Objects and rewrites two index structures
(Object.track_idx_2_obj_idx and PointTrackTable's two maps).  Getting any of
them wrong corrupts registration silently, so they are checked directly.
"""

import numpy as np
import pytest

from point2pose.data_types.point_track_table import PointTrackTable
from point2pose.modules.object.object import Object


def _make(n=20, obj_id=0):
    obj = Object(obj_id)
    tids = np.arange(n, dtype=np.int64)
    obj.add_key_points(
        new_key_points=np.random.default_rng(0).normal(size=(n, 3)),
        new_uncertainties=np.linspace(0.1, 1.0, n),
        new_valid=np.ones(n, dtype=bool),
        new_indices=tids,
        frame_id=0,
    )
    table = PointTrackTable.new(n0=n)
    table.add_new_points_to_track_obj_maps(tids, obj_id)
    return obj, table, tids


def _assert_index_consistent(obj):
    """Every kept track must map back to the row that actually holds it."""
    for row, tid in enumerate(obj.kp_track_indices):
        assert obj.track_idx_2_obj_idx[tid] == row, f"track {tid} -> wrong row"
    n = obj.key_points.shape[0]
    assert obj.kp_track_indices.shape[0] == n
    assert obj.uncertainties.shape[0] == n
    assert obj.valid.shape[0] == n
    assert obj.key_point_frames.shape[0] == n


def test_split_moves_rows_and_keeps_indices_consistent():
    obj, table, tids = _make(n=20)
    moving = tids[[3, 7, 11, 15]]
    keypoints_before = {int(t): obj.key_points[obj.track_idx_2_obj_idx[t]].copy()
                        for t in moving}

    child = obj.split_off(moving, new_id=1)
    assert child is not None
    assert child.key_points.shape[0] == 4
    assert obj.key_points.shape[0] == 16

    # the child carries the same geometry those tracks had on the parent
    for t in moving:
        row = child.track_idx_2_obj_idx[t]
        assert row >= 0
        np.testing.assert_allclose(child.key_points[row], keypoints_before[int(t)])

    _assert_index_consistent(obj)
    _assert_index_consistent(child)

    # parent must no longer claim the moved tracks
    for t in moving:
        assert obj.track_idx_2_obj_idx[t] == -1

    # no track is lost or duplicated across the two objects
    union = np.concatenate([obj.kp_track_indices, child.kp_track_indices])
    assert sorted(union.tolist()) == sorted(tids.tolist())


def test_track_table_ownership_transfers():
    obj, table, tids = _make(n=20)
    moving = tids[[3, 7, 11, 15]]
    child = obj.split_off(moving, new_id=1)

    moved = table.move_points_to_obj(child.kp_track_indices, 1)
    assert sorted(moved.tolist()) == sorted(moving.tolist())

    assert sorted(table.obj2track_map[1].tolist()) == sorted(moving.tolist())
    remaining = sorted(set(tids.tolist()) - set(moving.tolist()))
    assert sorted(table.obj2track_map[0].tolist()) == remaining

    # forward and reverse maps must agree
    for oid, idxs in table.obj2track_map.items():
        for t in idxs.tolist():
            assert table.track2obj_map[t] == oid


def test_child_inherits_parent_pose():
    obj, table, tids = _make(n=10)
    obj.pose = np.eye(4); obj.pose[:3, 3] = [0.1, 0.2, 0.3]
    child = obj.split_off(tids[[1, 2, 3]], new_id=1)
    np.testing.assert_allclose(child.pose, obj.pose)
    child.pose[:3, 3] += 0.5          # they must not share storage
    assert not np.allclose(child.pose, obj.pose)


def test_split_on_unknown_tracks_is_rejected():
    obj, table, tids = _make(n=10)
    assert obj.split_off(np.array([999, 1000]), new_id=1) is None
    assert obj.key_points.shape[0] == 10


def test_repeated_splits_partition_the_tracks():
    obj, table, tids = _make(n=30)
    a = obj.split_off(tids[0:5], new_id=1)
    table.move_points_to_obj(a.kp_track_indices, 1)
    b = obj.split_off(tids[5:12], new_id=2)
    table.move_points_to_obj(b.kp_track_indices, 2)

    for o in (obj, a, b):
        _assert_index_consistent(o)
    union = np.concatenate([obj.kp_track_indices, a.kp_track_indices, b.kp_track_indices])
    assert sorted(union.tolist()) == sorted(tids.tolist())
    assert len(set(union.tolist())) == 30

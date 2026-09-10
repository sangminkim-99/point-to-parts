import numpy as np
import pytest
from scripts.sim.compare_pose_traces import summarize, compare


def fixture():
    poses = np.tile(np.eye(4), (4, 2, 1, 1))
    return dict(frames=np.arange(4), part_ids=np.array([3, 7]), poses=poses,
                present=np.ones((4, 2), bool), observed=np.ones((4, 2), bool),
                votes=np.tile([[10], [0]], (4, 1, 1)), gt_names=np.array(['lid']),
                gt_poses=np.tile(np.eye(4), (4, 1, 1, 1)))


def test_common_subset_does_not_realign_and_missing_is_explicit():
    a, b = fixture(), fixture()
    a['poses'][1:, 0, 0, 3] = .1
    b['observed'][:2] = False
    report = compare({'a': a, 'b': b})['parts']['lid']
    assert report['common_frames'] == [2, 3]
    assert report['runs']['a']['common_median_mm'] == pytest.approx(100)
    assert report['runs']['b']['missing_frames'] == [0, 1]
    assert report['runs']['b']['coverage'] == .5


def test_mapping_uses_votes_not_best_pose_and_reports_switches():
    t = fixture()
    t['votes'][1, 1, 0] = 9  # duplicate, but do not choose the more accurate pose
    t['poses'][1, 0, 0, 3] = .2
    t['votes'][2:, 0, 0] = 0
    t['votes'][2:, 1, 0] = 10
    result = summarize(t)['lid']
    assert result['records'][1]['mm'] == pytest.approx(200)
    assert result['duplicate_candidate_frames'] == 1
    assert result['persistent_ids'] == [3, 7]
    assert result['id_switches_across_observations'] == 1


def test_absent_coverage_has_null_accuracy_and_mismatched_gt_rejected():
    a, b = fixture(), fixture()
    b['observed'][:] = False
    report = compare({'a': a, 'b': b})['parts']['lid']
    assert report['common_frames'] == []
    assert report['runs']['a']['common_median_mm'] is None
    b['gt_poses'][2, 0, 0, 3] = 1
    with pytest.raises(ValueError, match='GT trajectories'):
        compare({'a': a, 'b': b})

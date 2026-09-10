import numpy as np
from examples.multi_part.evaluate import save_pose_trace


def test_trace_keeps_deleted_ids_and_missing_truth_without_pickle(tmp_path):
    class Reader:
        def get_gt_pose(self, f, p):
            return np.eye(4) if f == 1 else None
    T = np.eye(4)
    gt = dict(parts=['body'], log=[(1, [T, T]), (2, [T])],
              id_log=[[0, 2], [0]], votes=[[[10], [6]], [[12]]],
              observed=[[True, False], [True]])
    path = tmp_path / 'trace.npz'
    save_pose_trace(Reader(), gt, path)
    d = np.load(path, allow_pickle=False)
    assert d['part_ids'].tolist() == [0, 2]
    assert d['present'].tolist() == [[True, True], [True, False]]
    assert np.isnan(d['poses'][1, 1]).all()
    assert np.isnan(d['gt_poses'][1, 0]).all()
    assert d['votes'][0, 1, 0] == 6
    assert not d['observed'][0, 1]

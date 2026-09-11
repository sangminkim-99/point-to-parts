import numpy as np
from scripts.sim.symmetry_metrics import surface_pose_errors


def test_off_origin_symmetry_keeps_strict_error_visible():
    points=np.array([[1.,0,0],[2.,0,0],[1.,1,0]])
    S=np.diag([-1.,-1,1,1]); S[:3,3]=[2,0,0]
    truth=np.eye(4)[None];truth[0,:3,3]=[.3,.1,1.]
    predicted=truth @ S
    result=surface_pose_errors(predicted,truth,points,[S])
    assert result['strict_rms_m'][0] > 1
    assert result['symmetry_rms_m'][0] < 1e-12
    assert result['symmetry_index'][0] == 1


def test_no_symmetry_never_hides_error():
    truth=np.eye(4)[None];predicted=truth.copy();predicted[0,0,3]=.1
    r=surface_pose_errors(predicted,truth,np.zeros((1,3)))
    np.testing.assert_allclose(r['strict_rms_m'],[.1])
    np.testing.assert_array_equal(r['strict_rms_m'],r['symmetry_rms_m'])

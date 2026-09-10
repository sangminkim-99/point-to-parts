import numpy as np
from scipy.spatial.transform import Rotation
from examples.multi_part.urdf_viser import covariance, load_export, ExportViewer
from examples.multi_part.author_state import save_snapshot
from test.object.test_author_demo import state


def test_gaussian_export_and_covariance(tmp_path):
    s = state()
    s.gaussian = dict(scales=np.array([[.01,.02,.03]]*2), quats=np.array([[1.,0,0,0]]*2), opacities=np.array([.8,.9]))
    save_snapshot(s,tmp_path/'export')
    data,joints,root = load_export(tmp_path/'export')
    np.testing.assert_array_equal(data['scales'],s.gaussian['scales'])
    np.testing.assert_allclose(covariance(data['scales'],data['quats'])[0],np.diag([.0001,.0004,.0009]))
    q = Rotation.from_euler('z',90,degrees=True).as_quat()[[3,0,1,2]][None]
    np.testing.assert_allclose(covariance(data['scales'][:1],q)[0],np.diag([.0004,.0001,.0009]),atol=1e-10)


def test_viewer_fk_uses_saved_root_pose(tmp_path):
    s=state(); save_snapshot(s,tmp_path/'export')
    data,joints,root=load_export(tmp_path/'export')
    v=ExportViewer.__new__(ExportViewer)
    v.base=s.parts[0].pose; v.joints=joints; v.root=root
    v.q=np.array([.3 if j['type']=='revolute' else 0. for j in joints])
    np.testing.assert_allclose(v.poses()['part1'],v.base @ s.parts[1].joint.at(.3),atol=2e-8)

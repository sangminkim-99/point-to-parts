from types import SimpleNamespace
import xml.etree.ElementTree as ET
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from examples.multi_part.urdf_export import export, joint_frames
from point2pose.pipeline.components.joint_model import JointModel


def model(kind):
    jm = JointModel()
    jm.kind = kind
    jm.A0 = np.eye(4)
    jm.A0[:3, :3] = Rotation.from_euler('xyz', [.3, -.5, .7]).as_matrix()
    jm.A0[:3, 3] = [.2, -.1, .4]
    jm._axis0 = np.array([1., 2., 3.]) / np.sqrt(14)
    jm._point0 = np.array([.1, -.2, .05]) if kind == 'revolute' else None
    jm.axis = jm.A0[:3, :3] @ jm._axis0
    jm.point = jm.A0[:3, :3] @ jm._point0 + jm.A0[:3, 3] if jm._point0 is not None else None
    jm.A = [jm.at(q) for q in [-.4, 0, .7]]
    return jm


def origin(joint):
    e = joint.find('origin')
    T = np.eye(4)
    T[:3, 3] = np.fromstring(e.attrib.get('xyz', '0 0 0'), sep=' ')
    T[:3, :3] = Rotation.from_euler('xyz', np.fromstring(e.attrib.get('rpy', '0 0 0'), sep=' ')).as_matrix()
    return T


def fk(xml, root_name, q):
    poses = {root_name: np.eye(4)}
    pending = list(xml.findall('joint'))
    while pending:
        progressed = False
        for joint in pending[:]:
            parent = joint.find('parent').attrib['link']
            if parent not in poses:
                continue
            T = origin(joint)
            M = np.eye(4)
            kind = joint.attrib['type']
            if kind != 'fixed':
                axis = np.fromstring(joint.find('axis').attrib['xyz'], sep=' ')
                v = q[joint.attrib['name']]
                if kind == 'revolute':
                    M[:3, :3] = Rotation.from_rotvec(axis * v).as_matrix()
                else:
                    M[:3, 3] = axis * v
            poses[joint.find('child').attrib['link']] = poses[parent] @ T @ M
            pending.remove(joint)
            progressed = True
        assert progressed, 'disconnected export'
    return poses


@pytest.mark.parametrize('kind', ['revolute', 'prismatic'])
def test_export_matches_fitted_motion_at_multiple_configurations(tmp_path, kind):
    jm = model(kind)
    parts = [SimpleNamespace(parent=0, joint=None), SimpleNamespace(parent=0, joint=jm)]
    path, _ = export(tmp_path / 'model.urdf', parts, lambda j: None)
    xml = ET.parse(path).getroot()
    for q in [-.4, -.13, 0, .23, .7]:
        actual = fk(xml, 'part0', {'j0_1': q})['part1']
        assert np.allclose(actual, jm.at(q), atol=2e-8)
    limit = xml.find("joint[@name='j0_1']/limit")
    assert np.isclose(float(limit.attrib['lower']), -.4)
    assert np.isclose(float(limit.attrib['upper']), .7)


def test_export_preserves_nonzero_root_and_chain(tmp_path):
    hinge, slide = model('revolute'), model('prismatic')
    parts = [SimpleNamespace(parent=2, joint=hinge), SimpleNamespace(parent=0, joint=slide),
             SimpleNamespace(parent=2, joint=None)]
    path, _ = export(tmp_path / 'chain.urdf', parts, lambda j: None)
    poses = fk(ET.parse(path).getroot(), 'part2', {'j2_0': .3, 'j0_1': .2})
    assert np.allclose(poses['part1'], hinge.at(.3) @ slide.at(.2), atol=2e-8)


def test_rigid_model_exports_fixed_transform(tmp_path):
    jm = model('rigid')
    parts = [SimpleNamespace(parent=0, joint=None), SimpleNamespace(parent=0, joint=jm)]
    path, _ = export(tmp_path / 'fixed.urdf', parts, lambda j: None)
    poses = fk(ET.parse(path).getroot(), 'part0', {})
    assert np.allclose(poses['part1'], jm.at(0), atol=2e-8)


@pytest.mark.parametrize('parents', [[0, 1], [0, 2, 1], [0, 9]])
def test_invalid_tree_is_rejected_before_writing(tmp_path, parents):
    parts = [SimpleNamespace(parent=p, joint=model('revolute')) for p in parents]
    with pytest.raises(ValueError):
        export(tmp_path / 'bad.urdf', parts, lambda j: None)
    assert not (tmp_path / 'bad.urdf').exists()


def test_viewer_respects_origin_rotation_fixed_offsets_and_joint_order(tmp_path):
    from examples.multi_part.urdf_view import read_urdf, link_poses
    hinge, slide = model('revolute'), model('prismatic')
    parts = [SimpleNamespace(parent=2, joint=hinge), SimpleNamespace(parent=0, joint=slide),
             SimpleNamespace(parent=2, joint=None)]
    path, _ = export(tmp_path / 'chain.urdf', parts, lambda j: None)
    _, joints, root = read_urdf(path)
    joints.reverse()  # child joints are now before their parents
    values = {'j2_0': .3, 'j0_1': .2}
    poses = link_poses(joints, root, [values.get(j['name'], 0.) for j in joints])
    assert np.allclose(poses['part1'], hinge.at(.3) @ slide.at(.2), atol=2e-8)

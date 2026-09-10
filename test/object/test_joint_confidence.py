import numpy as np
import pytest
from point2pose.pipeline.components.joint_model import JointModel


def model():
    jm = JointModel()
    jm.kind = 'prismatic'
    jm.axis = jm._axis0 = np.array([0., 0., 1.])
    jm.point = jm._point0 = np.zeros(3)
    jm.A0 = np.eye(4)
    A = np.tile(np.eye(4), (16, 1, 1))
    A[:, 2, 3] = np.linspace(0, .1, 16)
    return jm, A


def confidence(jm, A):
    return jm._confidence(A, np.zeros(len(A)), A[:, :3, 3],
                          [(0., 0., ('prismatic', jm.axis, jm.point))])


def test_symmetric_wide_axis_cone_is_not_reported_as_certain():
    jm, A = model()
    axes = iter([np.array([sign*.5, 0., np.sqrt(.75)]) for sign in [-1, 1]*8])
    jm._fit_prismatic = lambda *args: ('prismatic', next(axes), np.zeros(3))
    conf = confidence(jm, A)
    assert conf['axis_rms_deg'] == pytest.approx(30.)
    assert conf['axis_ok'] < .03
    assert conf['axis_bootstrap_samples'] == 16


def test_missing_bootstrap_evidence_is_not_perfect_axis_confidence():
    jm, A = model()
    jm._fit_prismatic = lambda *args: None
    conf = confidence(jm, A)
    assert conf['axis_ok'] == 0
    assert conf['conf'] == 0
    assert not conf['valid']


def test_axis_sign_is_not_uncertainty_and_clean_slide_stays_valid():
    jm, A = model()
    axes = iter([np.array([0., 0., sign]) for sign in [-1, 1]*8])
    jm._fit_prismatic = lambda *args: ('prismatic', next(axes), np.zeros(3))
    conf = confidence(jm, A)
    assert conf['axis_rms_deg'] == pytest.approx(0.)
    assert conf['valid']
    fitted = JointModel()
    for a in A:
        fitted.add(a)
    assert fitted.fit()
    assert fitted.kind == 'prismatic'
    assert fitted.confidence()['valid']
    assert fitted.confidence()['axis_ok'] == pytest.approx(1.)

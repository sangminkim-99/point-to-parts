from types import SimpleNamespace
import pytest
from examples.multi_part.naive import NaivePartTracker


@pytest.mark.parametrize('kind,valid,excitation,conf,ready', [
    ('rigid', True, 1., .9, False),
    ('revolute', False, 1., .9, False),
    ('revolute', True, .2, .9, False),
    ('revolute', True, 1., float('nan'), False),
    ('prismatic', True, 1., .9, True),
])
def test_joint_readiness_requires_valid_excited_articulation(kind, valid, excitation, conf, ready):
    tracker = SimpleNamespace(cfg=SimpleNamespace(joint_conf=.6), split_log=[])
    stats = dict(valid=valid, excitation=excitation, conf=conf, axis_std_deg=.2, span=.1)
    part = SimpleNamespace(joint=SimpleNamespace(kind=kind, confidence=lambda: stats))
    NaivePartTracker._note_controllable(tracker, 1, part, 20)
    assert hasattr(tracker, 'controllable') == ready

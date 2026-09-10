from types import SimpleNamespace as NS
from examples.multi_part.acquire import Acquisition


def scene():
    c=dict(conf=.9, valid=True, excitation=1., n=20, type_p=.99, axis_ok=.95)
    parent=NS(joint=None, observed=True, part_id=0, parent=0)
    child=NS(joint=NS(kind='revolute', confidence=lambda:c), observed=True, part_id=1, parent=0)
    return NS(parts=[parent,child]),c


def test_ready_revokes_on_lost_pose_and_reacquires_after_full_hold():
    s,c=scene();a=Acquisition(hold=2)
    a.step(0,s,1);a.step(1,s,1)
    assert a.ready and 'STABLE' in a.banner(s)[0]
    s.parts[1].observed=False;a.step(2,s,1)
    assert not a.ready and 'STABLE' not in a.banner(s)[0]
    assert a.ready_at == 0  # history remains distinct from current state
    s.parts[1].observed=True;a.step(3,s,1)
    assert not a.ready
    a.step(4,s,1);assert a.ready
    a.invalidate_tracking();assert not a.ready and 'lost' in a.banner(s)[0]


def test_identity_type_and_frame_gaps_reset_streak():
    s,c=scene();a=Acquisition(hold=2)
    a.step(0,s,1);s.parts[1].part_id=2;a.step(1,s,1)
    assert not a.ready
    s.parts[1].joint.kind='prismatic';a.step(2,s,1)
    assert not a.ready
    a.step(4,s,1);assert not a.ready
    a.step(5,s,1);assert a.ready


def test_unobserved_parent_and_nonarticulated_model_never_ready():
    s,c=scene();a=Acquisition(hold=1)
    s.parts[0].observed=False;a.step(0,s,1);assert not a.ready
    s.parts[0].observed=True;s.parts[1].joint.kind='rigid'
    a.step(1,s,1);assert not a.ready
    s.parts[1].joint.kind='revolute';c['conf']=float('nan')
    a.step(2,s,1);assert not a.ready

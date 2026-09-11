from examples.multi_part.evaluate import joint_summary


def test_duplicate_joint_matches_are_not_counted_as_distinct_coverage():
    rows=[dict(gt='rb2',ang=angle,kind='prismatic',ref_kind='prismatic',spec_kind='prismatic',dist=None) for angle in (.3,2.3)]
    s=joint_summary(rows)
    assert s['n']==2 and s['unique_gt_joints']==1 and s['duplicate_gt_rows']==1
    assert s['type_acc']==1.0


def test_joint_summary_separates_wrong_and_unknown_parent():
    rows=[dict(gt=f'rb{k}',ang=1.,kind='revolute',ref_kind='revolute',spec_kind='revolute',dist=.01,parent_matches_spec=value) for k,value in enumerate((True,False,None))]
    s=joint_summary(rows)
    assert s['correct_parent_rows']==1
    assert s['wrong_parent_rows']==1
    assert s['unknown_parent_rows']==1


def test_joint_axis_uses_parent_identity_after_reordering(monkeypatch):
    import numpy as np
    from types import SimpleNamespace as NS
    from examples.multi_part.evaluate import joint_metrics
    import point2pose.pipeline.components.joint_model as module
    class Ref:
        axis=np.array([1.,0,0]);kind='prismatic'
        def add(self,A): pass
        def fit(self): return True
    monkeypatch.setattr(module,'JointModel',Ref)
    reader=NS(joints=[dict(parent='base',child='lid',type='prismatic')],get_gt_pose=lambda i,name:np.eye(4))
    parts=[NS(part_id=10,parent=0,joint=None),NS(part_id=20,parent=0,joint=Ref())]
    wrong=np.eye(4);wrong[:2,:2]=[[0,-1],[1,0]]
    log=[(3,[wrong,np.eye(4)])] # parent 10 is at index 1 in this frame
    rows=[dict(part=0,gt='base'),dict(part=1,gt='lid')]
    result=joint_metrics(reader,[],NS(parts=parts),rows,log,ids_by_frame={3:[20,10]})
    assert len(result)==1 and result[0]['ang']==0.
    assert result[0]['identity_aligned']


def test_formatted_metrics_expose_duplicate_and_parent_status():
    from examples.multi_part.evaluate import fmt
    row=dict(gt='rb2',ang=1.,kind='prismatic',spec_kind='prismatic',ref_kind='prismatic',dist=None,parent_matches_spec=False,identity_aligned=True)
    text=fmt('example',dict(n_gt=3,covered=['rb2'],purity=.9,err={},joints=[row,row]),3)
    assert '1 unique GT joints' in text and '1 duplicate rows' in text
    assert 'parent_match=False' in text and 'identity_aligned=True' in text
    assert 'correct/wrong/unknown=0/2/0' in text

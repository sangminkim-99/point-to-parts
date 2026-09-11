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

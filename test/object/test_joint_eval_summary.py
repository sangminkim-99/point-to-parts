from examples.multi_part.evaluate import joint_summary


def test_duplicate_joint_matches_are_not_counted_as_distinct_coverage():
    rows=[dict(gt='rb2',ang=angle,kind='prismatic',ref_kind='prismatic',spec_kind='prismatic',dist=None) for angle in (.3,2.3)]
    s=joint_summary(rows)
    assert s['n']==2 and s['unique_gt_joints']==1 and s['duplicate_gt_rows']==1
    assert s['type_acc']==1.0

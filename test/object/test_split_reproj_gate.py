"""The alternative reprojection split gate (`residual_veto`, default off).

The default gate requires the child motion to REDUCE free-space contradiction by
0.08. Measured on a real drawer slide (capture 141006, f105-138): the child
motion RAISES contradiction (moved surface spills past the silhouette) so the
gain is negative and the slide is rejected -- while the on-object depth residual
drops and support rises. `residual_veto` accepts on that fit improvement and uses
reprojection only to veto genuine free-space contradiction, with a temporal
persistence requirement on the SAME partition.

These tests drive the decision helpers directly, so no depth/CUDA is needed.
"""
import numpy as np

from examples.multi_part.naive import NaiveConfig, NaivePartTracker


def gate(cfg_over=None):
    s = NaivePartTracker.__new__(NaivePartTracker)
    s.cfg = NaiveConfig()
    s.cfg.split_reprojection_mode = "residual_veto"
    for k, v in (cfg_over or {}).items():
        setattr(s.cfg, k, v)
    return s


def crow(sup_c, sup_s, fs_c, fs_s, pc=None, ps=None, npaired=0, off_c=0.0, off_s=0.0):
    """One child row of the shape `_accept_residual_veto` consumes."""
    return {"common": {"support": sup_c, "contradiction_freespace": fs_c,
                       "contradiction_offsilhouette": off_c},
            "separate": {"support": sup_s, "contradiction_freespace": fs_s,
                         "contradiction_offsilhouette": off_s},
            "paired_com": pc, "paired_sep": ps, "n_paired": npaired}


# a body child that barely moves: separate ~ common, no improvement
BODY = crow(0.70, 0.70, 0.02, 0.02, pc=0.005, ps=0.005, npaired=200)


# --------------------------------------------------------------- accept logic --

def test_default_mode_is_the_original_contradiction_gate():
    assert NaiveConfig().split_reprojection_mode == "contradiction"


def test_drawer_slide_accepted_on_paired_residual_improvement():
    # residual over shared samples drops 0.022 -> 0.004; extra contradiction is
    # OFF-silhouette (moved past the mask edge), free-space nearly flat
    drawer = crow(0.50, 0.65, 0.05, 0.06, pc=0.022, ps=0.004, npaired=120,
                  off_c=0.10, off_s=0.33)
    assert gate()._accept_residual_veto([BODY, drawer]) is True


def test_rigid_body_rejected_no_fit_improvement():
    rigid = crow(0.60, 0.60, 0.03, 0.03, pc=0.006, ps=0.006, npaired=150)
    assert gate()._accept_residual_veto([BODY, rigid]) is False


def test_motion_into_confirmed_free_space_is_vetoed():
    spurious = crow(0.55, 0.60, 0.05, 0.25, pc=0.020, ps=0.004, npaired=100)
    assert gate()._accept_residual_veto([BODY, spurious]) is False


def test_support_gain_alone_can_accept_when_residual_unavailable():
    drawer = crow(0.10, 0.45, 0.04, 0.05, pc=None, ps=None, npaired=0)
    assert gate()._accept_residual_veto([BODY, drawer]) is True


def test_unsupported_child_is_rejected():
    weak = crow(0.5, 0.10, 0.03, 0.04, pc=0.02, ps=0.004, npaired=100)  # sep.support 0.10<0.25
    assert gate()._accept_residual_veto([BODY, weak]) is False


def test_offsilhouette_spill_alone_does_not_veto():
    drawer = crow(0.50, 0.66, 0.05, 0.055, pc=0.020, ps=0.004, npaired=120,
                  off_c=0.10, off_s=0.40)   # offsil 0.10->0.40, free-space flat
    assert gate()._accept_residual_veto([BODY, drawer]) is True


# ------------------------------------------------- subset-shrink adversary --

def test_residual_branch_ignored_without_enough_paired_points():
    # a candidate that "improves residual" but only over a handful of paired
    # points must NOT pass on the residual branch (n_paired < min_points), and
    # here support does not rise either
    sneaky = crow(0.50, 0.50, 0.03, 0.03, pc=0.020, ps=0.002, npaired=5)
    assert gate()._accept_residual_veto([BODY, sneaky]) is False


def test_paired_residual_defeats_the_drop_hard_points_adversary():
    # enough paired points, but over the SHARED samples the residual does NOT
    # improve (0.019 -> 0.018); support flat too. The per-hypothesis residual
    # might have looked better by dropping points; the paired one does not.
    adversary = crow(0.55, 0.55, 0.03, 0.03, pc=0.019, ps=0.018, npaired=120)
    assert gate()._accept_residual_veto([BODY, adversary]) is False


# ----------------------------------------------------- persistence identity --

def groups(a, b):
    return [np.array(a), np.array(b)]


def test_persistence_rejects_a_single_frame_spike():
    s = gate({"split_reprojection_persist": 3})
    s.n = 62
    assert s._reproj_persist(0, True, groups(range(5), range(5, 10)), None) is False
    assert s._reproj_streak[0]["streak"] == 1


def test_persistence_accepts_a_sustained_same_partition():
    s = gate({"split_reprojection_persist": 3})
    g = groups(range(5), range(5, 10))
    for f, expect in [(50, False), (51, False), (52, True), (53, True)]:
        s.n = f + 1
        assert s._reproj_persist(0, True, g, None) is expect


def test_persistence_is_swap_invariant_across_frames():
    s = gate({"split_reprojection_persist": 2})
    s.n = 51
    assert s._reproj_persist(0, True, groups(range(5), range(5, 10)), None) is False
    s.n = 52                                   # same partition, children swapped
    assert s._reproj_persist(0, True, groups(range(5, 10), range(5)), None) is True


def test_persistence_resets_on_alternating_partitions():
    s = gate({"split_reprojection_persist": 3})
    s.n = 51; s._reproj_persist(0, True, groups(range(5), range(5, 10)), None)
    s.n = 52; s._reproj_persist(0, True, groups(range(10, 15), range(15, 20)), None)  # disjoint
    s.n = 53
    # a third, different partition still cannot reach 3 -- overlap keeps resetting
    assert s._reproj_persist(0, True, groups(range(20, 25), range(25, 30)), None) is False
    assert s._reproj_streak[0]["streak"] == 1


def test_persistence_ignores_a_duplicate_same_frame_call():
    s = gate({"split_reprojection_persist": 2})
    g = groups(range(5), range(5, 10))
    s.n = 51; assert s._reproj_persist(0, True, g, None) is False   # streak 1
    s.n = 52; assert s._reproj_persist(0, True, g, None) is True    # streak 2
    # a duplicate call on the SAME frame must not advance the streak further
    assert s._reproj_persist(0, True, g, None) is True
    assert s._reproj_streak[0]["streak"] == 2


def test_persistence_resets_on_a_frame_gap():
    s = gate({"split_reprojection_persist": 3})
    g = groups(range(5), range(5, 10))
    s.n = 51; s._reproj_persist(0, True, g, None)                   # streak 1
    s.n = 55; assert s._reproj_persist(0, True, g, None) is False   # gap>1 -> reset to 1
    assert s._reproj_streak[0]["streak"] == 1


def test_a_miss_resets_the_streak():
    s = gate({"split_reprojection_persist": 3})
    g = groups(range(5), range(5, 10))
    s.n = 51; s._reproj_persist(0, True, g, None)
    s.n = 52; assert s._reproj_persist(0, False, g, None) is False  # cond False -> reset 0
    assert s._reproj_streak[0]["streak"] == 0


def test_persist_one_is_the_per_frame_gate():
    s = gate({"split_reprojection_persist": 1})
    s.n = 10
    assert s._reproj_persist(0, True, groups(range(5), range(5, 10)), None) is True
    assert s._reproj_persist(0, False, groups(range(5), range(5, 10)), None) is False


def test_persistence_requires_both_children_not_only_stable_body():
    s = gate({"split_reprojection_persist": 3})
    for frame, lid in enumerate((range(100,110), range(110,120), range(120,130)), 50):
        s.n = frame + 1
        # The unchanged body used to produce mean Jaccard .5 even when
        # every moving-child track was replaced, passing the .5 threshold.
        assert not s._reproj_persist(0, True, groups(range(100), lid), None)
        assert s._reproj_streak[0]['streak'] == 1


def test_persistence_allows_partial_overlap_in_both_children():
    s = gate({"split_reprojection_persist": 2})
    s.n = 51
    assert not s._reproj_persist(0, True, groups(range(10), range(20,30)), None)
    s.n = 52
    assert s._reproj_persist(0, True, groups(range(22,32), range(2,12)), None)

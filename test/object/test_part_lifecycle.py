"""Instrumentation for newborn-part survival.

The failure these guard is take01: a part is born at frame 168 and merged
away at 170, having been judged on a q(t) that was back-filled from frames
before it existed. Final part count cannot see that -- both the run that
kept the part and the run that lost and re-found it end at two parts -- so
the tracker records identity, birth and evidence provenance instead.

Frame convention: hist entries and the merge check use the frame index
self.n - 1, and birth is stored in that same index. born is a separate
settle clock counted in self.n, which a merge resets.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from examples.multi_part.naive import NaiveConfig, NaivePart, NaivePartTracker
from point2pose.pipeline.components.joint_model import JointModel


class FakeJoint:
    """Only what the merge path touches, including add()'s rejection rule."""

    def __init__(self, n=0, smooth=0.5, reject=False):
        self.A = [np.eye(4) for _ in range(n)]
        self._smooth = smooth
        self._reject = reject
        self.kind = "revolute"

    def confidence(self):
        return {"smooth": self._smooth, "axis_std_deg": 1.0}

    def add(self, A):
        # mirrors JointModel.add: a degenerate transform is dropped silently
        A = np.asarray(A)
        if self._reject or not np.all(np.isfinite(A)) \
                or abs(np.linalg.det(A[:3, :3]) - 1) > 0.2:
            return
        self.A.append(A)


def tracker(parts, n, cfg=None):
    """A tracker with just enough state for the merge path."""
    s = NaivePartTracker.__new__(NaivePartTracker)
    s.cfg = cfg or NaiveConfig()
    s.parts, s.n, s.model = parts, n, None
    s.state = NaivePartTracker.SPLIT
    s.last_mode = ""
    s.alive_frames, s.seen_frames = {}, {}
    s.birth_log, s.death_log, s.merge_log = [], [], []
    s.split_log = []
    s._fit_box = lambda p: None
    s._reparent = lambda: None
    return s


def prov(frames, parent_id=0, source="live"):
    return [(f, parent_id, source) for f in frames]


def two_parts(joint, birth=168, obs_prov=None):
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0,
                      born=birth, birth=birth, joint=joint)
    child.obs_prov = obs_prov
    return [base, child]


# --------------------------------------------------------------- identity --

def test_merge_resets_born_but_not_birth():
    parts = two_parts(FakeJoint(20, smooth=0.01),
                      obs_prov=prov(range(150, 170)))
    s = tracker(parts, n=171)
    s._merge_rigid(170)
    assert len(s.parts) == 1
    assert s.parts[0].born == 171        # settle clock re-armed to self.n
    assert s.parts[0].birth == 0         # identity older than the merge


def test_a_split_continues_one_identity_and_births_the_other():
    # _accept hands the larger group the split part's part_id. That identity
    # did not end, so its birth must not be rewritten to the split frame.
    s = tracker([], n=0)
    s.next_part_id = 5
    s.co_same, s.co_seen = np.zeros((40, 40)), np.zeros((40, 40))
    s.track_born = np.zeros(40, np.int32)
    s.cfg.retro = False
    s.cfg.split_reprojection = False
    s._split_labels = lambda *a: None
    s.split_evidence = None

    parent = NaivePart(idx=np.arange(30), part_id=3, born=12, birth=11)
    s.parts = [parent]
    s.n = 169                            # i.e. processing frame index 168
    ok = s._accept(0, parent, [np.arange(20), np.arange(20, 30)],
                   [np.eye(4), np.eye(4)], None, None,
                   "coassoc", float("nan"), None,
                   sep_sigma=4.0, dbic=20.0)

    assert ok
    kept, made = s.parts[0], s.parts[1]
    assert kept.part_id == 3 and kept.birth == 11     # unchanged
    assert made.part_id == 5 and made.birth == 168    # frame index, not n
    assert {b["part_id"]: b["new_identity"] for b in s.birth_log} == {
        3: False, 5: True}
    assert s.birth_log[0]["frame"] == 168
    assert s.lifecycle_summary()["ids_born"] == 2     # part 0 and part 5


# ------------------------------------------------------------- provenance --

def test_provenance_records_only_observations_the_joint_accepted():
    # JointModel.add drops a degenerate transform silently. Counting
    # attempts rather than appends would overstate the evidence held.
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0, birth=100,
                      joint=FakeJoint())
    s = tracker([base, child], n=120)

    assert s._joint_add(child, np.eye(4), 110, "live") is True
    assert s._joint_add(child, np.full((4, 4), np.nan), 111, "live") is False
    assert s._joint_add(child, np.diag([5., 5., 5., 1.]), 112, "live") is False

    assert len(child.joint.A) == 1
    assert child.obs_prov == [(110, 0, "live")]
    assert s._evidence(child)["total"] == 1


def test_rebuild_counts_only_history_from_before_birth_as_retro():
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0,
                      born=169, birth=168, joint=FakeJoint())
    frames = list(range(160, 172))
    base.hist = [(f, np.eye(4), 0.0) for f in frames]
    child.hist = [(f, np.eye(4), 0.0) for f in frames]
    child.obs_prov = [(1, 1, "live")]    # must be replaced, not appended to

    s = tracker([base, child], n=172)
    s.root = 0
    s._rebuild_joint(1)

    ev = s._evidence(child)
    assert len(child.joint.A) == 12 and ev["total"] == 12
    assert ev["pre_birth"] == 8           # frames 160..167
    assert ev["rebuilt_post_birth"] == 4  # 168..171
    assert ev["earned"] == 0              # none of it was measured live


def test_rebuild_without_history_leaves_no_stale_provenance():
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0, birth=168,
                      joint=FakeJoint())
    child.obs_prov = [(1, 1, "live")] * 9
    s = tracker([base, child], n=171)
    s.root = 0
    s._rebuild_joint(1)                   # base.hist is None -> early return
    assert child.obs_prov == []
    assert s._evidence(child)["total"] == 0


def test_observations_against_a_former_parent_are_not_earned():
    # a reparent changes what the relative pose is relative to, so evidence
    # taken against the old parent is not evidence about this joint
    parts = two_parts(FakeJoint(6), birth=100,
                      obs_prov=prov(range(110, 113), parent_id=4)
                               + prov(range(113, 116), parent_id=0))
    s = tracker(parts, n=120)
    ev = s._evidence(parts[1])
    assert ev["total"] == 6
    assert ev["against_other_parent"] == 3
    assert ev["earned"] == 3


# ------------------------------------------------------------ merge ledger --

def test_death_ledger_separates_earned_evidence_from_retro():
    # the take01 shape: retro back-fill plus a few of the part's own
    parts = two_parts(FakeJoint(14, smooth=0.01),
                      obs_prov=prov(range(157, 168), source="rebuild")
                               + prov([168, 169, 170]))
    s = tracker(parts, n=171)
    s._merge_rigid(170)

    assert len(s.death_log) == 1
    d = s.death_log[0]
    assert d["part_id"] == 7 and d["birth"] == 168 and d["frame"] == 170
    assert d["lifetime"] == 2
    assert d["evidence"]["total"] == 14
    assert d["evidence"]["pre_birth"] == 11
    assert d["evidence"]["earned"] == 3
    assert d["earned_enough"] is False    # against merge_min_obs = 12
    assert d["absorbed_into"] == 0


def test_summary_counts_unearned_verdicts_whichever_way_they_fell():
    # a verdict reached before the part earned merge_min_obs is unsound even
    # when it happens to keep the part -- take01's part 2 survived that way
    merged = tracker(two_parts(FakeJoint(14, smooth=0.01),
                               obs_prov=prov(range(157, 168), source="rebuild")
                                        + prov([168, 169, 170])), n=171)
    merged._merge_rigid(170)
    assert merged.lifecycle_summary()["verdicts_on_unearned_evidence"] == {
        "merged": 1, "kept": 0, "total": 1}

    kept = tracker(two_parts(FakeJoint(56, smooth=0.66), birth=230,
                             obs_prov=prov(range(174, 229), source="rebuild")
                                      + prov([230])), n=231)
    kept._merge_rigid(230)
    assert len(kept.parts) == 2
    assert kept.lifecycle_summary()["verdicts_on_unearned_evidence"] == {
        "merged": 0, "kept": 1, "total": 1}


def test_a_part_that_earned_its_evidence_is_not_flagged():
    parts = two_parts(FakeJoint(20, smooth=0.01), birth=100,
                      obs_prov=prov(range(150, 170)))
    s = tracker(parts, n=171)
    s._merge_rigid(170)
    assert s.death_log[0]["evidence"]["earned"] == 20
    assert s.death_log[0]["earned_enough"] is True
    assert s.lifecycle_summary()[
        "verdicts_on_unearned_evidence"]["total"] == 0


def test_too_few_observations_records_null_not_nan():
    import json
    parts = two_parts(FakeJoint(3, smooth=0.01), obs_prov=prov([168, 169, 170]))
    s = tracker(parts, n=171)
    s._merge_rigid(170)
    assert len(s.parts) == 2
    m = s.merge_log[0]
    assert m["verdict"] == "too_few_obs"
    assert m["smooth"] is None and m["axis_std_deg"] is None
    json.dumps(s.lifecycle_summary(), allow_nan=False)   # must not raise


def test_a_driven_joint_is_recorded_as_kept_not_merged():
    parts = two_parts(FakeJoint(20, smooth=0.9), birth=100,
                      obs_prov=prov(range(150, 170)))
    s = tracker(parts, n=171)
    s._merge_rigid(170)
    assert len(s.parts) == 2 and s.death_log == []
    assert [m["verdict"] for m in s.merge_log] == ["kept_driven"]


# ----------------------------------------------------------------- summary --

def test_summary_distinguishes_survival_from_final_count():
    def birth(pid, frame, earned, pre):
        return {"frame": frame, "part_id": pid, "new_identity": True,
                "via": "coassoc", "n_points": 10, "hist": 0,
                "evidence": {"total": earned + pre, "earned": earned,
                             "pre_birth": pre, "rebuilt_post_birth": 0,
                             "against_other_parent": 0, "frames": []}}

    kept = tracker(two_parts(FakeJoint(20, smooth=0.9)), n=301)
    kept.alive_frames, kept.seen_frames = {0: 300, 7: 132}, {0: 300, 7: 120}
    kept.birth_log = [birth(7, 168, 20, 0)]
    a = kept.lifecycle_summary()

    churned = tracker(two_parts(FakeJoint(20, smooth=0.9)), n=301)
    churned.parts[1].part_id = 9
    churned.alive_frames = {0: 300, 7: 2, 9: 40}
    churned.seen_frames = {0: 300, 7: 2, 9: 30}
    churned.birth_log = [birth(7, 168, 2, 18), birth(9, 260, 20, 0)]
    churned.death_log = [
        {"frame": 170, "part_id": 7, "birth": 168, "lifetime": 2,
         "absorbed_into": 0, "earned_enough": False, "smooth": 0.01,
         "evidence": {"total": 20, "earned": 2, "pre_birth": 18,
                      "rebuilt_post_birth": 0, "against_other_parent": 0,
                      "frames": []}}]
    b = churned.lifecycle_summary()

    assert a["final_parts"] == b["final_parts"] == 2      # indistinguishable
    assert a["ids_died"] == 0 and b["ids_died"] == 1      # but not by identity
    assert a["ids_surviving"] == [0, 7]
    assert b["ids_surviving"] == [0, 9]

    row = {r["part_id"]: r for r in b["parts"]}[7]
    assert row["survived"] is False and row["death"] == 170
    assert row["evidence_at_death"]["earned"] == 2
    assert row["coverage"] == 1.0
    assert {r["part_id"]: r for r in a["parts"]}[7]["coverage"] == pytest.approx(
        120 / 132, rel=1e-3)


# ------------------------------------------------- against the real joint --

def test_provenance_matches_the_real_joint_models_acceptance_rule():
    # FakeJoint mirrors JointModel.add by hand; this pins that the mirror is
    # honest, because the whole earned/retro split rests on it.
    from point2pose.pipeline.components.joint_model import JointModel

    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0, birth=100,
                      joint=JointModel())
    s = tracker([base, child], n=120)

    good = np.eye(4)
    good[:3, 3] = [0.01, 0.0, 0.0]
    nonfinite = np.eye(4)
    nonfinite[0, 3] = np.inf
    scaled = np.eye(4)
    scaled[:3, :3] *= 2.0                # determinant 8, not a rotation

    # a rejected *historical* transform must not be counted as retro evidence
    assert s._joint_add(child, nonfinite, 90, "rebuild") is False
    assert s._joint_add(child, scaled, 91, "rebuild") is False
    assert s._joint_add(child, good, 92, "rebuild") is True
    assert s._joint_add(child, good, 110, "live") is True

    assert len(child.joint.A) == 2       # the real model dropped two
    assert len(child.obs_prov) == 2      # and provenance dropped the same two
    ev = s._evidence(child)
    assert ev["total"] == 2 and ev["pre_birth"] == 1 and ev["earned"] == 1


def test_birth_uses_the_frame_index_step_processes():
    # step() does `i = self.n; self.n += 1` and _fit appends history as
    # `self.n - 1`, so a birth recorded during that frame must equal i.
    # Driving the real step needs TAPIR and CUDA, so this mirrors its
    # prologue exactly and pins the invariant those two share.
    s = tracker([], n=0)
    s.next_part_id = 5
    s.co_same, s.co_seen = np.zeros((40, 40)), np.zeros((40, 40))
    s.track_born = np.zeros(40, np.int32)
    s.cfg.retro = False
    s.cfg.split_reprojection = False
    s._split_labels = lambda *a: None
    s.split_evidence = None
    parent = NaivePart(idx=np.arange(30), part_id=3, born=0, birth=0)
    s.parts = [parent]
    s.n = 40

    for _ in range(3):                   # three frames of step's prologue
        i = s.n
        s.n += 1
        hist_index = s.n - 1             # what _fit would append
        assert hist_index == i

    i = s.n
    s.n += 1
    s._accept(0, parent, [np.arange(20), np.arange(20, 30)],
              [np.eye(4), np.eye(4)], None, None,
              "coassoc", float("nan"), None, sep_sigma=4.0, dbic=20.0)

    made = s.parts[1]
    assert made.birth == i               # not s.n
    assert s.birth_log[-1]["frame"] == i
    # and the settle clock keeps its own, higher, counter
    assert made.born == i + 1 == s.n


# ---------------------------------------------------------------- probation --
#
# Probation (default OFF) withholds the merge verdict until a part has earned
# `probation_min_obs` genuine post-birth paired observations, then scores the
# merge on a joint fitted ONLY from those fresh observations -- never on the
# retro history back-fill produced. These tests use REAL JointModels and real
# relative-pose sequences, because the whole point is what the fresh model does
# with the part's own evidence, which FakeJoint cannot stand in for.

def _rot_z(theta):
    T = np.eye(4)
    c, s = np.cos(theta), np.sin(theta)
    T[:3, :3] = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    return T


def driven_revolute(n=14, step_deg=2.5):
    """A smooth revolute ramp: q(t) is linear, so smoothness is ~1 -- driven."""
    return [_rot_z(np.radians(step_deg * t)) for t in range(n)]


def white_jitter(n=14, seed=0, sig_deg=1.5, sig_t=0.006):
    """Zero-mean pose noise: q(t) is white, so smoothness is ~0 -- not driven."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        T = np.eye(4)
        T[:3, :3] = R.from_rotvec(np.radians(rng.normal(0, sig_deg, 3))).as_matrix()
        T[:3, 3] = rng.normal(0, sig_t, 3)
        out.append(T)
    return out


def real_jointed(seq, birth=100, source="live", parent_id=0, cfg=None):
    """A two-part tracker whose child carries a real JointModel filled from seq.

    Every observation is appended through `_joint_add`, so `obs_prov` and
    `joint.A` stay aligned exactly as they do in the live pipeline. Frames run
    from `birth`, and `source`/`parent_id` decide whether the evidence counts
    as earned.
    """
    cfg = cfg or NaiveConfig()
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0,
                      born=birth, birth=birth, joint=JointModel())
    s = tracker([base, child], n=birth + len(seq) + 1, cfg=cfg)
    for k, A in enumerate(seq):
        s._joint_add(child, A, birth + k, source)
    return s, child


def probation_cfg(min_obs=12):
    cfg = NaiveConfig()
    cfg.merge_probation = True
    cfg.probation_min_obs = min_obs
    return cfg


def test_probation_withholds_the_verdict_below_the_fresh_gate():
    # take01 part 1: 11 retro plus 3 of its own, judged and merged at frame 170.
    # With probation on, 3 < 12 earned, so the verdict is withheld, not reached.
    parts = two_parts(FakeJoint(14, smooth=0.01),
                      obs_prov=prov(range(157, 168), source="rebuild")
                               + prov([168, 169, 170]))
    s = tracker(parts, n=171, cfg=probation_cfg())
    s._merge_rigid(170)

    assert len(s.parts) == 2                 # not merged away
    assert s.death_log == []
    m = s.merge_log[0]
    assert m["verdict"] == "on_probation" and m["scored_on"] == "fresh"
    assert m["fresh_obs"] == 3               # what it has earned so far
    summ = s.lifecycle_summary()
    assert summ["probation_withheld"] == {"checks": 1, "part_ids": [7]}
    # a withheld check is a decision NOT to judge, so it is never an unsound verdict
    assert summ["verdicts_on_unearned_evidence"]["total"] == 0
    assert summ["config"]["merge_probation"] is True


def test_the_flag_alone_flips_the_take01_part1_outcome():
    # the same part and evidence; only merge_probation differs
    def run(prob):
        cfg = probation_cfg() if prob else NaiveConfig()
        parts = two_parts(FakeJoint(14, smooth=0.01),
                          obs_prov=prov(range(157, 168), source="rebuild")
                                   + prov([168, 169, 170]))
        s = tracker(parts, n=171, cfg=cfg)
        s._merge_rigid(170)
        return s

    off = run(False)
    assert len(off.parts) == 1 and off.death_log[0]["part_id"] == 7   # baseline merges
    on = run(True)
    assert len(on.parts) == 2 and on.death_log == []                 # probation keeps it alive
    assert on.merge_log[0]["verdict"] == "on_probation"


def test_probation_merges_when_the_fresh_evidence_is_white():
    # eligible by count, and the part's OWN observations are undriven jitter, so
    # the fresh-only model scores it as no joint and the split is undone
    s, child = real_jointed(white_jitter(14, seed=0), cfg=probation_cfg())
    s._merge_rigid(200)

    assert len(s.parts) == 1
    d = s.death_log[0]
    assert d["part_id"] == 7 and d["scored_on"] == "fresh" and d["fresh_obs"] == 14
    assert d["earned_enough"] is True        # 14 earned >= gate 12
    assert d["smooth"] < 0.15
    m = [x for x in s.merge_log if x["verdict"] == "merged"][0]
    assert m["scored_on"] == "fresh"


def test_probation_keeps_when_the_fresh_evidence_is_driven():
    # eligible by count, and the part's own observations trace a smooth joint,
    # so the fresh-only model keeps it
    s, child = real_jointed(driven_revolute(14), cfg=probation_cfg())
    s._merge_rigid(200)

    assert len(s.parts) == 2 and s.death_log == []
    m = s.merge_log[0]
    assert m["verdict"] == "kept_driven" and m["scored_on"] == "fresh"
    assert m["fresh_obs"] == 14 and m["smooth"] >= 0.15


def test_fresh_model_scores_only_earned_observations_not_retro():
    # the take01 part-2 trap: a joint whose FULL history is smooth because it was
    # back-filled, while the part's own fresh evidence is white. Probation must
    # judge on the latter, not the former.
    base = NaivePart(idx=np.arange(10), part_id=0)
    child = NaivePart(idx=np.arange(10, 20), part_id=7, parent=0,
                      born=200, birth=200, joint=JointModel())
    s = tracker([base, child], n=260)
    for k, A in enumerate(driven_revolute(30)):          # smooth, pre-birth, rebuilt
        s._joint_add(child, A, 160 + k, "rebuild")
    for k, A in enumerate(white_jitter(14, seed=0)):     # white, post-birth, live
        s._joint_add(child, A, 200 + k, "live")

    child.joint.axis_prior = 0.2
    child.joint.fit()
    assert child.joint.confidence()["smooth"] >= 0.15    # the full joint looks driven

    fm = s._fresh_model(child)
    assert len(fm.A) == 14                                # only the earned observations
    assert fm.confidence()["smooth"] < 0.15              # which are not driven
    assert s._evidence(child)["earned"] == 14
    assert s._evidence(child)["pre_birth"] == 30


def test_probation_never_merges_when_fresh_evidence_never_arrives():
    # boundedness: a part that keeps getting retro back-fill but earns little of
    # its own is checked again and again and merged never. It stays alive and
    # unverified -- it does not become credible by surviving.
    parts = two_parts(FakeJoint(33, smooth=0.01), birth=200,
                      obs_prov=prov(range(160, 190), source="rebuild")   # 30 retro
                               + prov([200, 201, 202]))                  # 3 earned
    s = tracker(parts, n=210, cfg=probation_cfg())
    for chk in (200, 210, 220, 230):
        s.n = chk + 1
        s._merge_rigid(chk)

    assert len(s.parts) == 2 and s.death_log == []
    summ = s.lifecycle_summary()
    assert summ["probation_withheld"] == {"checks": 4, "part_ids": [7]}
    assert summ["verdicts_on_unearned_evidence"]["total"] == 0


def test_probation_leaves_the_export_joint_untouched():
    # the fresh-only model is a diagnostic; scoring the merge must not refit or
    # replace the part's real tracking/export joint
    s, child = real_jointed(white_jitter(14, seed=0), cfg=probation_cfg())
    before = [A.copy() for A in child.joint.A]
    fm = s._fresh_model(child)
    assert fm is not child.joint
    assert len(child.joint.A) == len(before)
    assert all(np.array_equal(a, b) for a, b in zip(child.joint.A, before))

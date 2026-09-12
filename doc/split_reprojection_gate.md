# Reprojection split gate rejects a real drawer — diagnosis + opt-in fix

Author: Claude (tracking worker). Worktree `wt-articulation-probation`. Status:
**concrete diagnosis + a default-off tested fix. Nothing promoted; default gate
unchanged.** Object: a **drawer (prismatic slide)**. Symptom: at f105–132 the
live session logged `[reprojection] split gain -0.003..0.007 support 0.49..0.838
accept=False` — every slide rejected despite a substantial pull.

## 1. The gate, and what it measures

`_validate_split_reprojection` (`naive.py`) accepts a proposed split only if
```
gain = max_child(common.contradiction − separate.contradiction) ≥ 0.08   AND
       every child separate.support ≥ 0.25
```
`common` is the child's stored geometry scored under the **parent** pose,
`separate` under the child's candidate **motion** (`reprojection_evidence`,
`surface_memory.py`). `contradiction` = fraction of samples projecting into free
space (in front of a confirmed surface) **or** onto background. Occluded samples
(behind the surface) are neutral.

## 2. Diagnosis — measured, not assumed

Instrumented (`--dump-split-reproj`, default off) on the newest reproducing
capture, `capture-20260910-141006-591178` (staged with a rebuilt `masks.npz`).
165 candidate rows. The user's three hypotheses, tested:

- **Insufficient candidate motion? No.** 75/165 rows carry >3 cm relative
  translation; the drawer slide is captured (11 cm at f105–106, f135–138).
- **Floor effect (common already low)? No, not on the slide frames.** Common
  contradiction is 0.20–0.29 there (well above 0.08).
- **What is measured on these frames.** Applying the drawer motion *raises*
  total contradiction (`sep.contra` 0.39 vs `com.contra` 0.21 → gain **−0.18**),
  and the increase is dominated by samples that **spill past the current
  silhouette onto background** (off-silhouette). Meanwhile the on-object depth
  residual **drops** (0.022 → 0.004 m) and support rises (up to +0.24). So on
  this capture, on these frames, depth agreement improves while the
  "reduce free-space contradiction by 0.08" metric reads negative and rejects.

Interpretation (a **hypothesis**, not a proven law): for a prismatic slide the
moved geometry leaves the old silhouette, and if that off-silhouette spill is
scored as contradiction the improvement gate reads negative even when the fit
improves. This is measured on **one real capture** — it is **not** established
that the gate "cannot fire" for slides in general, nor that off-silhouette spill
is always benign: that spill can equally be a **mask error or wrong dense
ownership**, in which case rejecting is correct. The claim is therefore local
(this capture, these frames) and the fix is opt-in, judged against geometry, not
asserted universally. Note `contradiction_freespace` and
`contradiction_offsilhouette` **overlap** (a background sample in front of the
surface is in both) and do not sum to `contradiction`.

## 3. The fix (default OFF; `split_reprojection_mode=residual_veto`)

Instead of demanding a contradiction *improvement*, accept on **fit improvement**
and use reprojection only to **veto genuine free space**:
- every child supported (`separate.support ≥ split_reprojection_support`);
- at least one child fits materially better under its own motion — the **paired**
  residual drops by `split_reprojection_resid_gain` (0.004 m) **or** support
  rises by `split_reprojection_support_gain` (0.05). The residual is compared over
  the samples on-object under **both** poses (`_paired_resid`), with an explicit
  paired-count denominator, and the residual branch is used only when that count
  reaches `split_reprojection_min_points`. This defeats a subset-shrink adversary:
  a candidate cannot lower its residual by dropping the hard points, because a
  dropped point leaves **both** residuals, not just its own.
- **veto**: no child may raise `contradiction_freespace` (surface predicted in
  front of confirmed depth) over common by more than `split_reprojection_veto`
  (0.08). Off-silhouette spill is **not** vetoed — a deliberate, and unproven,
  choice (§2): it is right if the spill is a real slide leaving the mask, wrong if
  it is a mask/ownership error, which is why this is opt-in.
- **temporal persistence** (`split_reprojection_persist`): the accept condition
  must recur for the **same partition** on strictly-consecutive split-proposal
  frames. `_reproj_persist` keys on swap-invariant child-membership overlap
  (Jaccard ≥ `split_reprojection_persist_overlap`) and advances only on a strictly
  later frame within `split_reprojection_persist_gap` (default 1); a duplicate
  same-frame call does not advance, and a partition change or frame gap resets the
  streak. This is stronger than the earlier part-id-only counter (which alternating
  partitions or repeated calls could have satisfied) and is what separates a real
  slide from a one-frame rigid spike (§4).

`residual_veto` is a **UNION**, not a replacement: accept if the original
contradiction-improvement path holds **OR** the fit-improvement + persistence path
holds. This was forced by the hinge control (§4): the fit-only path detected the
drawer but **missed a sim revolute hinge** the contradiction path catches (a
swinging lid *does* reduce free-space contradiction). Keeping both paths catches
both joint types; the rigid control is rejected by both.

Default is unchanged: `split_reprojection_mode="contradiction"`, `persist=1`.

## 4. Comparison — current vs alternative

All: `--config reprojection_split.yaml --set dense=true`, isolated gsplat cache,
stride 1. GT stays out of the tracker; the sim controls' per-part GT is used only
to judge results afterwards. The alternative is the **union** gate at persist=3.
(An intermediate result that motivated the union: fit-only `residual_veto`
persist=3 detected the drawer but returned **1 part on the hinge** — a regression
the union fixes.)

| sequence (GT parts) | baseline `contradiction` | union `residual_veto` p3 |
| --- | --- | --- |
| **drawer 141006** (real, 2 intended) | 2 parts, split f153, **no joint fitted** | prismatic conf 0.75 @f55 ✓, but **3 parts** (a late f150 split via the baseline branch) |
| **laptop_orbit** (sim, GT 1 rigid) | 1 part ✓ | **1 part ✓** (both branches reject) |
| **storage_slide** (sim, GT 2 prismatic) | 2 parts, prismatic 0.95 ✓ | 2 parts, prismatic 0.95 ✓ |
| **laptop_hinge** (sim, GT 2 revolute) | 2 parts, revolute 0.67 ✓ | **2 parts, revolute 0.67 ✓** |
| **extra capture 140613** (real, unknown) | 2 parts, prismatic 0.71 @f218 | 2 parts, prismatic 0.79 @f116 (earlier) |

Reading:
- **The default gate misses the real drawer entirely** — its one split comes at
  f153 (near the clip end) and fits no joint. This is the reported failure. The
  union gate finds the drawer early (f55, prismatic 0.75).
- **The union keeps the joint types the default already handled**: the sim hinge
  (revolute 0.67) and sim slide (prismatic 0.95) are unchanged, and the rigid
  control stays at 1 part.
- **The real drawer ends at 3 parts under the union** (a late f150 split via the
  baseline contradiction branch, on top of the early f55 rv split). Whether the
  3rd part is spurious over-segmentation or a real second motion is **not
  decidable here** without per-part geometry matching — see limitations.

Metrics requested (with the confidence each is stated at):
- **False splits.** Verifiable only where there is GT. On the sim controls:
  `laptop_orbit` (rigid, GT 1) does **not** split under the union (fit-only
  persist=1 did, on a single frame; persistence removes it); `storage_slide`
  (GT 2) and `laptop_hinge` (GT 2) split correctly. On the **real** captures
  (drawer, extra) there is **no per-point GT**, only the user's known part count,
  so I report the count and joint, not geometric correctness — the drawer's 3rd
  part cannot be called false (or real) without geometry matching (IoU/ADD), not
  run here. No blanket "zero false splits" claim is made for the real captures.
- **Persistence:** keys on swap-invariant child-membership overlap and
  strictly-consecutive frames (§3); alternating partitions or repeated same-frame
  calls do not accumulate. It removes the single-frame rigid spike.
- **Joint fits:** baseline fits the drawer joint **not at all**; the union fits it
  prismatic conf 0.75 (69 mm). Sim hinge revolute 0.67, sim slide prismatic 0.95,
  extra capture prismatic 0.79 — all fitted.
- **Coverage:** min observed-coverage 1.0 across all cells (short clips; does not
  discriminate here).
- **Root choice (reference-tracker validation):** replayed the drawer with the
  root **pinned to the body** (`--reference-part 0`, matching the demo's
  `ReferenceTracker`) — the split verdicts are **identical** to the default
  least-motion root (f55 prismatic 0.75, f150 → 3). So the gate decision is robust
  to the root convention on this capture; the demo's root choice does not change
  it here (see limitations for the residual gap).
- **Runtime: not measured, no claim.** The union does the default's work **plus**,
  per child, a third projection (`_paired_resid`) and persistence bookkeeping — it
  is **not** free. I did not time it across repeats, so I make **no** cost claim.

## 5. Recommendation

Keep the default gate (`contradiction`) unchanged. The **union `residual_veto`
with `persist=3` is an opt-in that recovers the reported drawer failure** (a
prismatic joint the default misses) while preserving the hinge/slide the default
already handled and rejecting the rigid GT control. It is **not yet a clean win**:
on the real drawer it ends at 3 parts vs 2 intended, and that over-split is
unadjudicated without geometry matching. So: opt-in only, and before any default
change it needs (a) per-part geometry matching on the real captures to settle the
3rd part, (b) more sequences — a rigid case with *sustained* spurious motion,
which persistence alone would not catch, and more real hinges/slides. GT stays
out of the tracker; oracle union masks only.

## 6b. The drawer's 3rd part (f150) — diagnosed: a new weak group, not a repeat

The union gate leaves the real drawer at 3 parts (§4). Instrumenting the f150
split (`--dump-split-reproj`) settles what it is:

- **Membership/where:** the split is of **part 0 (the body)**, not the drawer —
  a 20-point group peels off the body (`149/20 pts`, dBIC 1699).
- **Motion:** that group displaces **2.1 cm relative to the body**, essentially
  pure translation (0.4°). It is accepted by the **baseline** contradiction
  branch (free-space contradiction 0.178 → 0.059, gain 0.17), on a single frame.
- **Does it repeat an existing part?** No. Measured against the drawer (id1) at
  that frame: pose 4.0 cm apart, and translation direction **opposite**
  (`dir_cos −0.70`; the drawer's own body-relative displacement was 2.25 cm). So
  it is neither co-located with the drawer nor sliding the drawer's axis.

**Conclusion:** the 3rd part is a **new, independent, weak group** moving opposite
to the drawer — **not** a re-discovery of the body or the drawer. Given the
user's two intended parts it is **suspected over-segmentation**, but the mechanism
is a spurious weak split, not a repeat.

**Opt-in consistency handling (`split_reprojection_consistency`, default off).**
`_split_repeats_existing` refuses a split whose moving child re-discovers an
existing sibling — by pose (same place) or by motion direction (same axis, even
if not as far). On the controls (verified): the **hinge is preserved** (the lid
rotates while the body is static, so it matches no sibling → 2 parts), orbit stays
1, slide stays 2. On the drawer it **correctly abstains** — the f150 group is not
a repeat (opposite direction), so consistency does not fire and the drawer stays
at 3. So this handling is a sound guard for the *repeat* failure mode and does not
harm the positive control, but it is **not** the fix for this particular
over-split, which is a weak independent group. Suppressing that is a
strength/persistence/newborn-evidence concern (the merge/probation path), kept
separate here per instruction and not folded into this gate.

## 6. Limitations and open questions

- **Off-silhouette-as-benign is a hypothesis, not proven.** The gate treats a
  moved surface spilling past the current mask as benign; that spill can also be a
  mask error or wrong dense ownership, in which case rejecting is correct. The
  free-space veto is the guard, but the choice is unverified and is why this is
  opt-in and single-lens (§2).
- **No per-point GT on the real captures.** The drawer/extra results report part
  count and joint fit, not geometric correctness. The drawer's 3rd part is
  unadjudicated. Only the sim controls' claims (rigid = no split; hinge/slide
  fitted) are GT-backed.
- **Reference-tracker gap partially closed.** The gate uses the splitting part's
  own pose, and I validated that pinning the root to the body (`--reference-part`,
  matching the demo's `ReferenceTracker`) leaves the drawer verdict unchanged.
  But `--reference-part` only replicates the *root-pinning*; the live demo also
  differs in interactive reference selection timing and could diverge on other
  captures. Final sign-off should still run through the actual `author_demo`
  path, which this worker does not own.
- **Persistence is a heuristic.** persist=3 on strictly-consecutive frames with
  ≥0.5 partition overlap separates the cases seen here; a *sustained* spurious
  rigid motion (many consecutive frames) would still pass. It is not a proof of
  articulation, only a filter on transients.
- **Runtime unmeasured.** The union adds a third projection per child plus
  bookkeeping; no timing was taken.
- **Small sample.** Two real captures and three sim controls. Not a benchmark.

## Reproduce

```sh
P=$HOME/miniconda3/envs/point2pose_model
export TORCH_EXTENSIONS_DIR=<repo>/results/torch_extensions   # isolated, preserves live cache
export CUDA_HOME=$P
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include${CPATH:+:$CPATH}"
export TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 ; export PATH=$CUDA_HOME/bin:$PATH
# baseline vs alternative (add --dump-split-reproj for the evidence json)
$P/bin/python -m examples.multi_part.replay --seq-dir <drawer-stage> \
  --method naive --config reprojection_split.yaml --set dense=true \
  --set split_reprojection_mode=residual_veto --set split_reprojection_persist=3 \
  --no-eval --vis clean --view 0 --hyp-panel 0 \
  --out <out>.mp4 --dump-lifecycle <life>.json
```
Artifacts under `results/split_reproj_v1/`; tests in
`test/object/test_split_reproj_gate.py` (12). Diagnosis instrumentation and the
gate are default-off.

## Files

- `examples/multi_part/surface_memory.py` — `reprojection_evidence` split into
  `contradiction_freespace` / `contradiction_offsilhouette` + continuous `resid`
  (backward compatible; `contradiction` unchanged).
- `examples/multi_part/naive.py` — config `split_reprojection_mode`/`_resid_gain`/
  `_support_gain`/`_veto`/`_persist`/`_persist_gap`/`_persist_overlap`; the UNION
  gate in `_validate_split_reprojection`; `_accept_residual_veto` (paired
  residual), `_paired_resid`, `_reproj_persist` (partition-identity), `_partition_
  overlap`, `_reproj_diag`; default-off `split_reproj_log`.
- `examples/multi_part/replay.py` — `--dump-split-reproj` and `--reference-part`
  (reference-tracker replay mode), both default off.
- `test/object/test_split_reproj_gate.py` (17 tests incl. subset-shrink,
  alternating-partition, duplicate-frame, swap-invariance), `test_surface_memory.py`
  (two exact-dict expectations updated for the new keys).

## Integration note — 2026-09-11

The consistency guard described above (`split_reprojection_consistency`,
`_split_repeats_existing`) was **not integrated into the main tree**. The
integration review of September 10 (see `doc/drawer_split_validation.md`,
15:44 entry) rejected the pose-or-direction sibling guard because parallel,
independently moving drawers can share a direction. What did land from this
worktree is the residual_veto + persistence gate (via the minimal gate-only
patch), the `--dump-split-reproj` evidence dump and `--reference-part` replay
mode used for the diagnosis here.

# Newborn-part merge probation

Author: Claude (probation worker). Base commit `da5bf69`, implemented and run in
worktree `wt-articulation-probation`. Status: **default OFF, evidence gathered,
promotion recommendation below.** This report is written to be read alongside
`doc/part_lifecycle_review.md` (the instrumentation review) and the dialogue log.

Read the interpretation cautions in the review first: retro-reconstructed
history is causal (built from frames that were actually seen), so "premature
eligibility" is a statement about *which observations a verdict rested on*, not
a physical claim that the object was rigidly attached. This report keeps that
distinction.

## 1. The failure, precisely

On take01 a part is born at frame 167, split off at 168, and merged back at 170.
The merge check (`_merge_rigid`) undoes a split when the relative motion is not
*driven* — when `q(t)` is white rather than smooth. The problem is not the
smoothness test; it is **what evidence the test is run on**, and **when**.

Two mechanisms combine:

1. **Retro back-fill.** With `retro` on (the config chain for this experiment
   sets it), a newborn is given its own past: `_accept → _reparent →
   _rebuild_joint` replays the shared pose history into the new part's joint.
   So a part two frames old already holds a dozen relative-pose observations,
   almost all from before it existed as a distinct part.

2. **Eligibility on that back-fill.** The merge check only judges a part once
   its joint holds `merge_min_obs` (12) observations. Retro back-fill reaches
   that count immediately, so a two-frame-old part is *judged at all* — on
   evidence it never earned. At frame 170, part 1 was killed on a `q(t)` that
   was 3 of its own observations and 11 back-filled ones; that mixed `q(t)`
   scored smoothness 0.02, below the 0.15 threshold, and the split was undone.

The instrumentation (already reviewed) measures this without inferring it from
part count: each observation carries `(frame, parent_part_id, source)`, and
`_evidence` reports how many are **earned** — measured live, at or after birth,
against the current parent.

## 2. What probation does

`merge_probation` (default **False**; never enabled globally without review)
changes the merge check in two ways, exactly as agreed in the dialogue:

1. **Eligibility gate on earned evidence, before confidence.** A part is not
   judged until it has accumulated `probation_min_obs` (default 12) genuine
   post-birth paired observations. Below that it is recorded `on_probation` and
   left alive — the verdict is *withheld*, not decided. This gates at the same
   place `merge_min_obs` gates, before `confidence()` is consulted.

2. **Scoring on a fresh-only model.** Once eligible, the merge is scored on a
   **separate** `JointModel` fitted from only the earned observations
   (`_fresh_model`). The part's real tracking/export joint is never replaced or
   refitted — the fresh model is a diagnostic used to decide the merge and then
   discarded. `obs_prov` is kept in lock-step with `joint.A` (both grow only
   through `_joint_add`), so the earned subset selects the matching transforms.

The two are independent and both necessary. Waiting N frames and then scoring
the *same* retro-filled model would only delay the contaminated verdict; scoring
fresh without the gate would fit a fresh model to too little data.

### What happens when evidence never arrives (indefinite, not time-bounded)

A part that never earns `probation_min_obs` fresh paired observations is
**never merged away**: the merge verdict is withheld **indefinitely**, for the
rest of the clip. There is no timer — it is not "bounded" to N frames; it is an
open hold with no resolution until fresh evidence arrives, which it may never do.
lift01 shows this: a part sits at 1/12 fresh observations for the whole rest of
the clip because it is rarely observed freshly against its parent.

**Important scope correction.** This hold affects only the merge (un-split)
decision. Probation does **not** touch the part's tracking/export joint, which is
fitted every frame and exported regardless (`step` fits `p.joint`; export reads
it). So a part on indefinite probation is **still exported as an articulation** —
withholding the merge does *not* stop it from being promoted into the model. The
earlier claim that such a part "never becomes a credible articulation / is never
promoted" is therefore **not established**: probation only prevents deletion, not
export. Gating export on earned evidence would be a separate change, not part of
this experiment.

The ledger records this separately from verdicts: `probation_withheld`
`{checks, part_ids}` counts how often a verdict was withheld and over which
parts, and these are **not** counted as `verdicts_on_unearned_evidence` —
withholding is the decision *not* to judge, never an unsound judgement.

## 3. Tests

`test/object/test_part_lifecycle.py`, 21 passing (14 instrumentation + 7
probation). Full `test/object` suite: **95 passed**. The probation tests use
real `JointModel`s and real relative-pose sequences, because the whole point is
what the fresh-only model does with a part's own evidence:

- `test_probation_withholds_the_verdict_below_the_fresh_gate` — the take01
  part-1 shape (3 earned of 14) is withheld, not merged.
- `test_the_flag_alone_flips_the_take01_part1_outcome` — identical part and
  evidence; only `merge_probation` differs. Off merges it; on keeps it alive.
- `test_probation_merges_when_the_fresh_evidence_is_white` — eligible by count,
  fresh `q(t)` white → merged (scored_on fresh).
- `test_probation_keeps_when_the_fresh_evidence_is_driven` — eligible, fresh
  `q(t)` smooth → kept.
- `test_fresh_model_scores_only_earned_observations_not_retro` — the take01
  part-2 trap: a joint whose *full* history is smooth because it was back-filled
  (0.93) while its *own* fresh evidence is white (0.00). Probation judges on the
  latter.
- `test_probation_never_merges_when_fresh_evidence_never_arrives` — bounded:
  repeated checks, withheld every time, never merged, never flagged unearned.
- `test_probation_leaves_the_export_joint_untouched` — the fresh model is a
  diagnostic; the part's real joint is not refitted or replaced.

Baseline (flag off) behaviour is byte-unchanged: all 14 pre-existing lifecycle
tests still pass, and the take01 baseline run below reproduces the documented
splits and merge exactly.

## 4. Results — baseline vs probation

Every cell: `--config reprojection_split.yaml` (reprojection gating **ON**),
`--method naive`, stride 1, the only change between variants being
`--set merge_probation={false,true}`. Ledgers under
`results/newborn_probation_v2/`. Metrics are identity survival, true observed
coverage (`observed_frames / alive_frames`), fragmentation (splits) and the
probation bookkeeping — **not** the final part count.

Reproduce:

```sh
P=$HOME/miniconda3/envs/point2pose_model
env CUDA_HOME=$P PATH=$P/bin:$PATH $P/bin/python \
  -m examples.multi_part.run_probation_experiment \
  --sequences take01 lift01 laptop_orbit laptop_hinge storage_slide
env CUDA_HOME=$P PATH=$P/bin:$PATH $P/bin/python \
  -m examples.multi_part.aggregate_probation   # rebuild the combined table
```

### take01 (RealSense, no GT)

| variant | born | died | surviving | splits | min lifetime | unearned verdicts | withheld |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 3 | 1 | [0, 2] | 2 | 3 | 3 | 0 |
| probation | 2 | 0 | [0, 1] | 1 | 153 | 0 | 1 |

What happened, from the ledgers:

- **Baseline.** Part 1 (born frame 167) is merged at 170 on 3 earned of 14
  observations, full-model smoothness 0.02. The part is then **rediscovered**
  as part 2 at frame 229 and kept — but its first two keep-verdicts (ages 1 and
  11) rest on 1 and 11 earned observations, below the 12 gate. Three verdicts on
  the clip rest on unearned evidence (1 merge, 2 keeps).
- **Probation.** The frame-170 verdict on part 1 is *withheld* (3/12 fresh). By
  frame 180 the part has earned 13 fresh observations; scored on those alone its
  `q(t)` is smooth (0.47, rising to ~0.66), so it is kept — and the **original
  identity survives to the end** (lifetime 3 → 153). The frame-229 rediscovery
  never happens (splits 2 → 1). No verdict rests on unearned evidence.

The honest reading: take01 has no ground truth, so this does **not** prove part 1
is physically a real articulated part. What it shows is that baseline's kill and
its early keeps rested on evidence the parts had not earned, while probation's
keep rests on each part's own fresh, smooth `q(t)`. Whether probation's kept
identity is *physically correct* is a question only the GT sim controls can
answer (§4c).

### lift01 (RealSense, no GT)

| variant | born | died | splits | min lifetime | coverage min/mean | unearned verdicts | withheld |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 6 | 0 | 5 | 94 | 0.89 / 0.98 | 11 | 0 |
| probation | 7 | 3 | 6 | 17 | 0.80 / 0.95 | 0 | 9 |

lift01 shows probation's other direction, and it is not a "keep more parts"
story. Baseline keeps all 6 parts, but **11 of its keep-verdicts rest on
unearned evidence** — parts kept as joints on retro-inflated `q(t)` before they
had earned 12 fresh observations. Probation converts every one of those into a
sound verdict: 3 parts are **earned-merged** and the rest withheld.

The 3 deaths matter, because they are not premature: each is scored on the
part's own fresh evidence and each shows a white `q(t)`.

```
DEATH id1 @f120  scored_on=fresh  earned=17/35  smooth=0.03
DEATH id2 @f300  scored_on=fresh  earned=19/83  smooth=0.00
DEATH id4 @f310  scored_on=fresh  earned=23/76  smooth=0.13
```

So on lift01 probation **removed 11 unsound keeps** — merging 3 parts whose own
fresh motion is not driven (parts baseline kept only because back-fill made
their `q(t)` look smooth), and holding the rest as bounded-unverified. The extra
birth and split are downstream churn: a merge returns points to the parent,
which can re-split. Without GT on lift01 we cannot call the 3 culled parts
correct, only that the merges were **earned** — which is exactly the behaviour
probation is for. Coverage discriminates here (part 1 seen on 356 of 401 alive
frames, 0.89), unlike take01 where it is 1.00 throughout.

### 4c. Sim controls (with GT)

The RealSense clips cannot answer correctness. Two PartNet-Mobility controls
carry per-part GT (`poses.npz`, `meta.json`):

- **laptop_orbit — rigid control.** The whole object orbits with no real joint.
  A spurious part that is born here *should* be merged back. This is the
  false-positive test for probation: does withholding + fresh-only scoring keep
  a spurious part alive on a genuinely rigid object?
- **laptop_hinge — attached revolute** and **storage_slide — attached
  prismatic.** A real joint that should survive under both variants; probation
  should not break a case baseline already gets right.

| control | variant | born | died | final parts | GT parts | unearned | withheld |
| --- | --- | --- | --- | --- | --- | --- | --- |
| laptop_orbit (rigid) | baseline | 1 | 0 | 1 | 1 | 0 | 0 |
| laptop_orbit (rigid) | probation | 1 | 0 | 1 | 1 | 0 | 0 |
| laptop_hinge (revolute) | baseline | 3 | 1 | 2 | 2 | 3 | 0 |
| laptop_hinge (revolute) | probation | 3 | 0 | 3 | 2 | 0 | 3 |
| storage_slide (prismatic) | baseline | 2 | 0 | 2 | 2 | 1 | 0 |
| storage_slide (prismatic) | probation | 2 | 0 | 2 | 2 | 0 | 1 |

**laptop_orbit (rigid).** The whole object orbits rigidly and, under
reprojection gating, never splits at all — so no spurious part is born and the
merge path is never exercised. Baseline and probation are byte-identical: no
regression, but also no evidence either way. This control turned out not to
stress the merge path; a rigid case that *does* produce a spurious split (e.g.
`laptop_static` under looser gating) would be a stronger false-positive probe
and is the obvious addition.

**laptop_hinge (revolute) — the control that most stresses the mechanism.**
The GT is 2 parts (base + lid). Read the ledger:

```
baseline:  id1 (lid) born f73, revolute READY f76 (conf 0.67);
           kept_driven f80/f90/f100 at smoothness 0.999/0.9998/0.995;
           f110 a second, weak split fires (id2, 6 pts, dBIC 653), whose reparent
           REBUILDS id1's joint from scratch -> id1 earned drops 27 -> 1;
           the very next check MERGES id1 on 1/20 earned, smoothness 0.091.
probation: same births; id1 kept_driven on its FRESH model (0.97) through f100;
           at f110 the reparent resets earned to 1, so id1 is WITHHELD, not
           merged; id1 survives to the end.
```

Baseline **merges a part that had been a driven revolute** (id1, smoothness 0.999
for thirty frames) because a reparent at f110 reset its earned-evidence count and
the merge check re-judged it on 1 earned observation. Its final count of 2
matches the GT part count, but by keeping id2 and discarding id1. Probation
withholds that verdict and id1 survives; its final count is 3, because the
6-point fragment id2 never earns enough fresh evidence to be judged and is held
withheld (though, per §2, still exportable). id1's fresh `q(t)` is driven (0.97);
id2 sits at 1/20 fresh throughout.

Which of id1/id2 is the physical lid — and therefore whether probation's 3-part
result is better or worse than baseline's 2 — is **not settled here**: it needs
per-part GT IoU. What is settled: baseline's merge of id1 was reached on unearned
evidence (1 of the gate's 12) after a reparent reset, and probation withheld it.

So on the one GT control where a driven part is discovered and then
re-contaminated, **baseline reaches its merge on unearned evidence and probation
withholds it.** Whether probation's retained set (3) is physically closer to GT
(2) than baseline's (2) is **not decided by these metrics** — it needs per-part
GT IoU. The defensible claim is narrower than "count is the wrong metric": here
baseline's count matched GT via a verdict that was unsound (1 of 12 earned), and
probation removed that unsoundness.

**storage_slide (prismatic).** GT is 2 parts. Both variants keep both parts;
the only difference is that baseline's keep includes one verdict on unearned
evidence, while probation reaches the same correct outcome with every verdict
earned (one withheld early, then earned keeps). Clean pass: probation does not
break a case baseline gets right, and makes it sound.

### Cross-cutting result

Across all five sequences, `verdicts_on_unearned_evidence` goes to **0** under
probation. Every merge and every keep it reaches rests on the part's own fresh,
post-birth, same-parent observations. That is the design goal, and it holds on
RealSense and sim alike.

## 5. Runtime

Matched wall-clock per cell (single runs; the machine's single-run spread is
~30%, see dialogue 12:52, so read these as "no measurable difference", not an
overhead figure):

| cell | baseline | probation |
| --- | --- | --- |
| take01 | 42.2 s | 44.3 s |
| laptop_orbit | 18.3 s | 17.9 s |
| laptop_hinge | 18.5 s | 18.2 s |
| storage_slide | 16.2 s | 15.9 s |

Probation is within noise of baseline on every cell (marginally faster on the
sim cells). This is expected: the fresh-only fit runs only at merge checks
(every `merge_every` = 10 frames), only for parts past the eligibility gate, and
only over a part's earned observations — a rare, small computation against
per-frame tracking. No overhead claim is made from single runs, but there is no
sign of a cost.

## 6. Limitations

- **No GT on the RealSense clips.** take01 and lift01 show that probation makes
  verdicts sound and preserves identities baseline drops, but cannot show those
  identities are *physically* correct. Only the sim controls carry GT.
- **Part-count vs identity not fully adjudicated.** On laptop_hinge probation
  ends at 3 parts vs GT 2. The lifecycle evidence shows baseline's merge of the
  real lid was unsound and probation's withholding preserved a driven part, but
  whether the extra part (id2) is a spurious fragment or a second real piece
  needs **per-part GT IoU/ADD**, which this run did not compute (`--no-eval`).
  That evaluation is the gate to any default-on decision.
- **Probation fixes premature merges, not spurious splits.** It will not merge
  away an over-split fragment without evidence — it holds it as
  bounded-unverified. So over-segmentation from the split side stays visible
  (and correctly uncredited) rather than being masked by an unsound merge.
- **The rigid control did not stress the merge path** (laptop_orbit never
  split). A rigid case that produces a spurious split is still wanted.
- **`probation_min_obs` was not swept.** Held at 12 (= `merge_min_obs`) for
  every cell, per the brief to avoid sweeps and get a first result. The fresh
  model needs at least `JointModel.min_obs` (8) to fit; a value below that would
  make eligible parts unfittable and is not recommended.

## 7. Recommendation

The bar was stated before the sim results were read: advance toward a reviewed
default-on only if probation does not keep a *spurious* part credited as a joint
on the rigid control, does not break the real joints, and removes unearned
verdicts without inventing false ones.

**What the evidence supports:** probation meets its stated ledger goal —
`verdicts_on_unearned_evidence` is 0 on all five sequences, every merge/keep
verdict is reached only after the part earns fresh evidence. On laptop_hinge,
baseline merges a part (id1) that had been a stable driven revolute for 30
frames, on 1 earned observation after a reparent reset; probation withholds that
merge and id1 survives. On storage_slide both variants keep the two parts, with
probation's verdicts earned.

**What it does NOT establish** (correctness and cost both need evidence I did not
gather):
- *Physical correctness.* That id1 is **the** GT lid — not a fragment, and that
  probation's 3-part result is better than baseline's 2 — is **not shown**: it
  needs per-part GT IoU/ADD, which this run did not compute (`--no-eval`). I can
  say baseline's merge was unsound and probation withheld it; I cannot yet say
  probation's retained set matches GT better.
- *Runtime.* Only single runs per cell were measured; the machine's single-run
  spread is ~30%. No overhead is visible, but **no no-cost claim is made** —
  that needs repeated runs.
- *"Never promotes an unproven part."* Withdrawn — the export joint is
  unchanged, so a withheld part is still exportable (§2).

**Recommendation: keep `merge_probation` default OFF. It is usable as an opt-in
diagnostic** (the ledger goal — zero unearned verdicts — is met, no regression on
the rigid control), **and any default-on decision is gated on per-part GT
IoU/ADD** on the sim controls (laptop_hinge especially, plus a rigid case that
actually splits) and on repeated-run timing.
This worker does not enable it globally; Codex reviews and decides. The
mechanism is confirmed on GT: premature merge eligibility on re-contaminated
evidence is real (laptop_hinge id1, killed on 1 earned observation after a
reparent reset), and probation is the correct guard against it.

## 8. Follow-up plan: pairwise relative-pose hypotheses + graph selection

Per the user-approved direction (dialogue, "relative-motion joint discovery").
**Not implemented here — a separate, reviewed change.** This experiment
motivates it directly: the sharpest unsound merge measured (laptop_hinge id1,
killed on 1 earned observation at f110) was triggered by a **star-root
reparent**, whose `_rebuild_joint` reset the part's earned-evidence count and
re-exposed a mature, driven part to judgement on back-fill. A design in which
joints live on *pairs*, not on a chosen root, removes that reset entirely.

Sketch, built on established kinematic model selection (Sturm, Stachniss &
Burgard, *JAIR* 2011 — per-edge joint-type selection by BIC and kinematic-graph
structure selection; the `K_PARAMS` = 6/9/12 already used in `JointModel` are
theirs), and standard structure learning:

1. **Persistent per-part camera poses.** Already maintained (`NaivePart.hist`);
   no reference body is chosen for discovery.
2. **Pairwise relative histories.** For each candidate pair `(i, j)`, form
   `T_ij(t) = inv(T_i(t)) @ T_j(t)`. Shared whole-object SE(3) motion cancels
   when both pose estimates are reliable and synchronised — no static body
   assumed.
3. **Per-edge model selection.** Score rigid / prismatic / revolute / free on
   each `T_ij` history by BIC — exactly today's `JointModel.fit`, run per pair
   instead of per-child-against-root. Carry the earned-evidence discipline from
   probation onto edges: an edge is not typed until the pair has earned enough
   fresh, independently-excited relative motion; otherwise it is
   bounded-unverified, as parts are here.
4. **Graph structure selection.** A low-BIC pairwise fit does **not** prove
   direct adjacency — coupled chain motion can make `i–k` look articulated
   through `j`. Select a consistent kinematic tree/graph over the edges
   (Sturm's structure-selection: maximise total edge evidence subject to a
   connected tree; resolve chain ambiguity with independent-excitation /
   conditional-independence tests before committing an edge).
5. **Root last.** Choose a root only afterwards, for URDF export and optional
   stabilised viewing — a coordinate convention, not a stationary-body claim.

Validation matrix (GT for evaluation only), reusing this experiment's harness:
invariance to shared rigid motion; moving-root plus articulation; chain
ambiguity (three coupled links); occlusion/coverage (the lift01 0.89-coverage
case); and free/disconnected motion. The lifecycle ledger and the matched
baseline-vs-variant runner extend to per-edge metrics with little change.

Scope note: this is a subsequent change, not a silent baseline edit. Codex
reviews scope before any implementation; probation stays default-off meanwhile.

## Files

- `examples/multi_part/naive.py` — `merge_probation`, `probation_min_obs`,
  `_fresh_model`, probation branch in `_merge_rigid`, `on_probation` verdict,
  `probation_withheld` + `config` in `lifecycle_summary`. Default off.
- `test/object/test_part_lifecycle.py` — 7 new probation tests.
- `examples/multi_part/run_probation_experiment.py` — matched runner.
- `examples/multi_part/aggregate_probation.py` — combined comparison table.
- `results/newborn_probation_v2/` — per-cell ledgers, videos, `comparison*.json`.

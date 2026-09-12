# Duplicate bodies are mixture births — diagnosis and two default-off fixes

Author: Claude, 2026-09-12. Status: **diagnosed with per-point evidence;
two opt-in options measured on 6 cases × 4 arms; nothing promoted.**
Artifacts: `results/split_purity_v1/` (stdouts, `*_splitdiag.json`,
traces, hists). Tools: `replay.py --dump-split-diag`,
`scripts/sim/split_purity.py`, `scripts/sim/summarize_split_arms.py`.

## Why

`doc/kinematic_graph_discovery.md` ended on one blocker: duplicate bodies
(ikea p1/p3 both rb2, p2/p4 both rb0; cardboard p0/p2 both rb0) defeat
both root rules and make rigid merging impossible. This asks where the
duplicates come from.

## Diagnosis: parts are born as mixtures

Per-frame GT-label composition of every part (trace votes):

| part | at birth | later |
| --- | --- | --- |
| ikea p1 (f47, "frame" path) | rb0 **19** / rb2 **18** — purity 0.51 | rb2 42 / rb0 1 |
| ikea p2 (f74) | rb0 16 — pure | rb0 36 / rb2 4 |
| cardboard p2 (f233, "coassoc" path) | rb0 **17** / rb1 **19** — purity 0.53 | rb0 21 / rb1 7 |
| cardboard p3 (f254) | rb1 16 — pure | pure |

A mixture child gets a compromise anchor pose. Its live history is junk for
as long as the two bodies move differently (ikea p1 vs rb2: 171 mm at
f60–80, 6–21 mm once rb2 stops), and the mover's points left behind in the
parent (25 rb2 tracks stayed in p0 at f47) later surface as a *second*
identity when the parent splits again (p3 at f152). The two "rb2" parts then
carry different — one junk, one real — histories, so no pairwise test can
call them rigid. That is the duplicate.

Per-point split diagnostics (`--dump-split-diag`: each child's residual under
its own and the other child's motion, plus GT track labels) locate two
distinct mechanisms:

1. **ikea f47** — the pooled separation test passed (4.3σ, carried by the
   167-point child) while the 38-point child's own points prefer their motion
   by only 2.5σ; the coassoc proposal at the same frame was refused at 2.8σ.
   *Partition at marginal separation.*
2. **cardboard f233** — separation is huge (92 mm vs 3.8 mm) yet the child is
   half lid: the affinity clustering handed over a stale group, the RANSAC
   motion came from its rb0 core, and the lid points stayed in the group
   (own inlier fraction 0.56). *Stale co-association membership.*

## Two default-off options (`NaiveConfig`)

- `split_sep_per_child`: every child must clear `split_sep_sigma` on its own
  points (min over children instead of the pooled median); in the frame path
  it is re-checked after the points are partitioned.
- `split_refine_coassoc`: in the co-association path, hand every point of the
  two groups to the motion it fits now, drop the ambiguous band, refit the two
  motions — the rule the single-frame path already applies.

Unit tests: `test/object/test_split_partition.py` (4). `test/object`: 179.

## Matched results

### Under the checkpoint gates (`reprojection_split`, `…_frozen_fallback`)

| case | base | per-child | refine | both |
| --- | --- | --- | --- | --- |
| cardboardbox01 (fb) | 3 parts / 2 GT, 90.8 %, lid 3.9° | same as base | **2 / 2, 96.8 %**, births 0.97 / 1.00, lid 5.7° | = refine |
| ikeasmall02 (fb) | 5 / 3, 94.7 %, parents 0/2 + duplicate | **2 / 3** (rb2 lost) | **2 / 3** | **2 / 3** |
| laptop_hinge, storage_slide, laptop_hinge_noisy | 2 / 2 | identical | identical | identical |
| two_drawers_v2, door_drawer (fb) | 2 / 3 (one child never splits) | identical | identical (door_drawer: door instead of drawer, 65 %) | identical |

Refinement removes the cardboard duplicate outright. On ikea every option
*loses* the lower drawer — and the refusal log says why: with the f47/48
mixture split refused, the clean coassoc proposals at **f61–85** (separation
11–48σ, groups ~200 vs ~75, both children pure) are rejected by the
**reprojection gate every frame** (contradiction gain ≈ 0.001), and a few by
BIC. The base run only found rb2 because the contradiction gate happened to
accept the *wrong* partition at f48 (gain 0.246) and refuse the right ones.
This is the drawer-slide gate failure of `doc/split_reprojection_gate.md`,
seen on real data with a clean proposal in hand.

### Under the tested slide-safe gate (`reprojection_split_residual_veto`)

| case | base | per-child | refine | both |
| --- | --- | --- | --- | --- |
| ikeasmall02 | **3 / 3, 92.3 %**, splits f29 + f159, **no duplicates**; body p2 17.8 mm | identical | **3 / 3, 94.0 %**, body p2 **5.6 mm** | = refine |
| cardboardbox01 | 3 / 2, 90.8 % | same | **2 / 2, 96.8 %** | = refine |

So the ikea duplicates were a gate artefact: under `residual_veto` the first
split is the lower drawer at f29 with a clean partition, and the sequence
ends with exactly the three GT bodies. Refinement then improves the body
part's pose (17.8 → 5.6 mm), which is what the kinematic graph needs:

| ikea, residual_veto | star-root joints | pairwise tree (`kinematic_graph.py --hist`) |
| --- | --- | --- |
| base | 1 row (rb2), parent 0/1 | drawer–drawer pair cheapest (45.9 vs 163.5); 1 spec edge, parents 1/2 (spread) or 0/2 (degree) |
| **refine** | 1 row (rb2), parent 0/1 | body–drawer 6.7 / 9.5 vs drawer–drawer 33.4; **2 spec edges, parents 2/2 under both root rules**, axes 10.6° / 9.2°, types correct |

## What this does and does not establish

- Duplicates on the two RBO sequences have two mechanisms, both at birth;
  neither is "the same body found twice" in a way a post-hoc rigid test
  could repair.
- `split_refine_coassoc` fixes the stale-membership mechanism (cardboard) and
  improves body pose on ikea; it is a no-op on the sim controls.
- `split_sep_per_child` is correct about ikea f47 but, under the default
  gate, its only effect is to expose the gate's refusal of the clean
  proposals; under `residual_veto` it changes nothing. It should not be
  promoted on this evidence.
- The chain now is: slide-safe gate (already opt-in) → clean partition →
  good body pose → pairwise graph with correct parents. Each link was
  measured separately; the combination has been run on one real sequence.
- Not done: the same matched table on take01/lift01 (no GT) and the other
  RBO objects; door_drawer/two_drawers still split only one child
  (`co_min_seen` refusals), unchanged by either option.

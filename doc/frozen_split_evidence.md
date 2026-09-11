# Preserve geometry for split validation

The live map can add the moved surface and carve away its old position before
part discovery finishes. This weakens contradiction-based split evidence.

The opt-in `reprojection_split_frozen.yaml` retains up to 1,536 part-local
Gaussian means per part at its first observed `over > 0` frame, after sparse
pose fitting and before that frame's dense maintenance. Split validation uses
these means with the current parent and candidate child poses, current depth,
and current object mask. The working map still grows, carves and renders.
No future observations, GT poses, or GT part labels enter this snapshot.

Snapshots persist across quiet frames, because refreshing them could erase the
same evidence again. Any change in the ordered persistent part IDs clears all
snapshots before new ones can be captured. This conservatively invalidates
ownership after split/merge/reordering. Storage is bounded by part count and
sample limit. A false early motion alarm can retain stale or incomplete
geometry; this option therefore remains off by default. Hidden geometry still
cannot supply positive support; the existing occlusion-aware gate is unchanged.

## Evidence and limits

Claude's matched experiment in `results/growth_ablation_v1/` used recording
`capture-20260910-165946-036912`. Baseline split at frame 305; no-growth plus
no-carve split at 181. A read-only shadow comparison of the same 96 baseline
candidates accepted 58 with frozen geometry versus 1 with live geometry, first
at frame 181 versus 305. These are gate decisions on one unlabeled recording,
not segmentation accuracy. They implicate combined map maintenance; they do
not isolate growth and carve's individual causal contributions.

The worker's snapshot was captured after the trigger frame's full step. This
implementation captures before dense maintenance and resets after topology
changes. Thus those replay numbers are motivation, not validation numbers for
this implementation. Unit tests cover immutability, memory bounds, causal
triggering, held observations and topology invalidation. Matched replay and
rigid-motion false-split controls remain required before changing defaults.

Run from the repository root:

```bash
CONFIG=reprojection_split_frozen.yaml PORT=8090 OUT=results/author_demo_live \
  examples/multi_part/run_author_demo.sh live --depth 1
```

Logs report `geometry=frozen@<frame>` when retained evidence is in use.

## Implementation validation at 619ed71 (September 10)

The exact shipped implementation has now been replayed by the tracking worker
in an isolated checkout. Detailed commands/logs are in
`results/frozen_gate_v1/frozen_gate_validation.md`.

| Recording | Baseline split frames | Frozen split frames | Final parts baseline / frozen |
| --- | --- | --- | --- |
| drawer stage_165946 | 305 | 181 | 2 / 2 |
| box stage_164102 | 38, 103, 163; merge 130 | 44 | 3 / 2 |

The box first split is six frames later; its f1 snapshot scored only 0.077 at
f38 against a 0.080 threshold. Another later proposal lacked enough group
geometry. These are real tradeoffs of preserving incomplete early geometry.
No segmentation GT exists for these recordings. Fewer parts or fewer deaths
alone does not establish correctness. Recorded step medians were 97.7/92.7 ms
(drawer baseline/frozen), 170.3/121.5 ms (box); single shared-machine runs and
changing part counts prevent a speedup claim.

Separate oracle-union-mask simulation controls (rigid turnover and vertical
lift) reported zero false splits in both configurations and matching pose
errors. Crucially, no split proposal reached the frozen gate, so these do not
stress-test its rejection of spurious proposals. Turnover still re-locks with
approximately 180-degree observed-pose error; freezing split geometry does not
solve opposite-face tracking. Growth counts differ slightly between variants
without a resolved explanation; no deterministic equivalence claim is made.

A regression test now directly checks that replacing the live geometry cannot
erase the retained gate evidence, and that disabling the option returns to the
live-model decision. Keep the option experimental until noisy rigid controls
actually exercise the gate and coverage/ownership errors are evaluated.

### Noisy rigid controls (reviewed September 11)

The demo worker's `doc/gaussian_growth_probe.md` in its worktree records three
noisy/occluded slab controls plus rigid moving_laptop. Slabs still generated no
split proposals. Moving_laptop generated one at f50: live gain 0.001 and frozen
snapshot f6 gain 0.002, both rejected. This exercises the gate on one negative
candidate, not a broad false-positive guarantee. Artifacts are under
`results/thin_turnover_v1/*_gate/growth_probe.json`.

Noisy turnover remained observed through the flip with roughly 180-degree
pose error. A guard triggered only after an unobserved frame cannot address
this failure. Do not conflate stable part count with correct pose or promote
relock abstention based solely on the clean control.

### Additional RealSense recordings at 88505c1

The worker compared `/home/smkim/data/mp/take01` and `lift01` using existing
prepared binary masks. These are **not RBO sequences**, despite the legacy
artifact filename `results/frozen_gate_v1/frozen_gate_rbo_validation.md`.
Neither recording provides GT for the reported comparison.

| Recording | First split baseline / frozen | Identity deaths baseline / frozen | Final parts baseline / frozen |
| --- | --- | --- | --- |
| take01 | 168 / 172 | 1 / 0 | 2 / 3 |
| lift01 | 104 / 84 | 0 / 0 | 6 / 4 |

On take01 the baseline's split-168/merge-170 churn disappears, but a small
additional part is created at f250. Joint-kind estimates differ between the
arms; fewer deaths and earlier splits are not proof of correct articulation.
On lift01 minimum observed-frame coverage increases from 0.89 to 1.00, but
observed status is not independently verified pose accuracy. Step medians are
97.4/102.3 ms and 170.7/154.3 ms respectively (single runs).

Actual RBO validation with existing predicted masks has been requested
separately for ikeasmall02_o and cardboardbox01_o. Defaults remain unchanged.

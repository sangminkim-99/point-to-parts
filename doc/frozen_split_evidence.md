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

# Drawer split validation — active experiment

User reports substantial relative sliding motion, with reprojection split gain
approximately -0.003 to 0.007 and support approximately 0.49 to 0.84 over frames
105–132. Current support threshold is 0.25 and required contradiction reduction
is 0.08. These candidates pass support but fail improvement. This establishes
the rejection condition, not whether candidate motions are physically correct.

## Experiment ownership

Claude tracking worker: instrument candidate motion, sparse residual improvement,
common/separate reprojection evidence and persistent-group history; implement an
opt-in alternate gate and run matched controls. Claude demo worker: identify
recent drawer captures and validate completeness without changing originals.
Codex: review evidence, test alternate gate and integrate only a tested checkpoint.

## Required comparisons

1. Current gate, unchanged, on a complete drawer recording.
2. Alternative gate on the identical frames, seed and tracking configuration.
3. Rigid whole-object motion negative control with identical gate settings.
4. Simulated prismatic control; GT only for evaluation, oracle union masks named.

A temporally consistent multi-motion explanation may supply positive evidence
when depth-contradiction reduction is uninformative. Reprojection must still
reject unsupported/contradictory candidates. Do not merely set gain to zero:
that would accept ties without adding independent evidence. The planar-motion
explanation remains a hypothesis until common/separate evidence is inspected.
Both coassociation and frame-fallback proposal paths must be covered; their
existing sparse validation differs.

Report split frame, persistent identities, false splits on the rigid control,
observed-frame coverage, retained geometry, joint type/axis/range where estimable,
and runtime. Part count alone is insufficient. Retain the old default until
matched evidence warrants promotion. Sparse tracking and joint-state evidence
must remain causal. Newly accepted split must not immediately merge away.

Recent recordings: capture-20260910-140613-170537 (318 RGB/masks),
140806-284484 (144), 140847-131720 (272). At inspection 141006-591178
had 180 RGB and 179 masks; treat as potentially still writing and snapshot only
complete matched RGB/depth/mask frames for any offline evaluation.

All worker gsplat execution must use a separate persistent TORCH_EXTENSIONS_DIR
and stable CUDA_HOME, CPATH, and TORCH_CUDA_ARCH_LIST=8.9. User cache untouched.

## Manager review of first candidate (September 10, 14:43)

Worker comparison reports split f153 -> f55 and fitted prismatic joint for
residual_veto with persistence=3. This is an initial behavioral result, not
validated geometry or joint accuracy on the real recording. Per-part GT is not
available. No default promotion.

Review found persistence keyed only by parent ID, allowing changing candidate
partitions; same-frame calls also advance the streak. Independently reproduced
three calls at frame 100 returning False, False, True. Worker must require
distinct frames and consistent child membership/motion, with permutation handling.
Continuous residual comparison also uses different on-object subsets under each
hypothesis: require a fair sample comparison or guard against improved means
caused by dropping difficult samples. Off-silhouette disagreement is not by
itself benign; stale/wrong ownership and mask errors remain possible.

Requested replay through the actual ReferenceTracker demo path, because ordinary
NaivePartTracker replay can choose a different root and change the outcome.
Required next controls include a second complete capture and a hinge, in addition
to existing rigid and simulated drawer controls. Worker is implementing fixes.

## September 10, 15:44 — integration review

V2 worker fixed same-frame/partition persistence and uses paired residuals.
Its union with the original contradiction branch retains the hinge control but
permits a late third group in the real drawer. Worker diagnosis assigns that
group to a small body subset moving differently from the drawer. This is not
proof of a third physical part. A proposed pose-or-direction sibling guard is
not accepted: parallel independently moving drawers can share direction.
Requested a minimal opt-in patch without that guard and without unrelated
probation changes, plus clean-HEAD validation and an explicit experimental config.

Gaussian diagnostic UI review found that reference freshness was based on any
observed part, observed RGB was passed an annotated overlay, and a proposed
support fraction counted background valid depth without residual agreement.
Requested corrections and an actual GPU smoke test before integration. The
per-part nearest-depth image must be labeled approximate compositing, not a
globally alpha-composited Gaussian render. Show stale frames and render errors.
No new code defaults or claimed performance improvement promoted in this review.

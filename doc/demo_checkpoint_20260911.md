# Interactive articulation demo checkpoint — September 11, 2026

## Run

From the repository root, experimental live configuration:

```bash
CONFIG=reprojection_split_frozen_fallback.yaml \
PORT=8090 OUT=results/author_demo_live \
examples/multi_part/run_author_demo.sh live --depth 1
```

Fixed camera, move the object; Save exports model.npz and a URDF when a
supported connected joint tree exists. New exports preserve Gaussian scales,
quaternions and opacity. View a saved export without a camera or CUDA build:

```bash
~/miniconda3/envs/point2pose_model/bin/python \
  -m examples.multi_part.urdf_viser \
  results/author_demo_live/<saved-directory> --port 8091
```

Viewer opens at saved tracked poses. Switch to URDF joint poses for sliders;
fitted poses can differ from tracking by the model residual. Gaussian browser
rendering and point previews are selectable. Legacy exports are point-only.
The viewer remains stopped at the user's request.

## Confirmed progress

- Frozen split evidence with live fallback for insufficient retained support:
  RealSense drawer first split 305→181 frames. Actual RBO cardboard recovered
  baseline splits 191/234/255 after frozen-only failed entirely. This remains
  opt-in: incomplete early geometry, extra parts and incorrect graphs remain.
- Native Viser Gaussian diagnostics, per-part dock views, render reuse and
  depth-supported RGB error. This is not render-loss pose optimization.
- Mesh-free URDF/model viewer. SAPIEN agrees with viewer FK at 34 configurations
  of one saved multi-joint export (max matrix-entry difference 3.73e-7).
- Identity-aligned RBO cardboard fallback matched lid joint: 3.900 degrees
  axis error, 7.1 mm axis-line distance, correct mapped parent. Reference axis
  is fit to mocap with the same joint estimator; no robot-readiness claim.
- Joint evaluator now exposes duplicates, parent agreement, persistent IDs.
- Final object test suite: 159 passed. Meaningful code checkpoints pushed.

## Material limitations / failed hypotheses

- RBO ikea still has 5 predicted parts vs 3 GT. Two matched rows claim rb2;
  both have wrong mapped parents despite small axis angles. Full graph is wrong.
- Thin-slab opposite-face tracking can settle at 180-degree error. Existing
  incremental/continuity/keyframe variants fail on the tested symmetric slab.
- Relock guard blocks wrong updates but holds indefinitely; noisy turnover
  can stay falsely observed, bypassing a guard that requires a prior loss.
- Appearance error detects inconsistency for distinct faces but paired color
  scoring still favors the wrong pose over the oracle in tested controls.
  Never-seen backside geometry/appearance is absent from the stored model.
- Gaussian duplication during rigid translation persists. Growth-scale repair
  improves footprint, not underlying pose/map consistency.
- Sparse orphan reattachment is experimental; dense historical ownership is
  not fully solved. No successful end-to-end robot manipulation demonstrated.
- RBO masks are SAM2-propagated with a GT-derived initialization box; simulated
  controls use oracle union masks. Neither is fully annotation-free evaluation.

## Evidence

- results/frozen_gate_rbo_v1/frozen_fallback_validation.md
- results/frozen_gate_rbo_v1/rbo_joint_identity_rerun.md
- results/frozen_gate_rbo_v1/ikea_joint_identity_rerun.md
- results/thin_turnover_diag_v1/turnover_config_ablation.md
- results/thin_turnover_diag_v1/keyframe_baseline_turnover.md
- results/thin_turnover_v1/appearance2_rigid_distinct/appearance_probe.json
- results/urdf_viewer_validation/saved_box_independent.json

Large artifacts are ignored locally, not included in Git. Unrelated pre-existing
uncommitted lifecycle/simulation edits and user-owned directories are preserved.

## Next work after this scheduled development window

Prioritize correct parent graph and pose/geometry consistency over more gates.
Evaluate visible-surface orientation and multiple plausible poses through
turnover; report uncertainty when observations cannot distinguish them. Test
render-based scoring only with correctly posed/visible surface memory and fair
coverage. Validate learned joint trajectories and manipulation geometry beyond
matched axis statistics. No novelty claim is established by the current tests.

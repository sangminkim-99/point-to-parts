# Online articulation → URDF: progress and continuation

Updated 2026-09-10. User authorized continued work through tomorrow, prefers
established methods before inventing new ones, and requested frequent commits and
pushes. A thread heartbeat `articulation-urdf` is scheduled hourly until
2026-09-11 18:00 America/New_York. Continue substantive work, not just status checks.
The repository is `/home/smkim/workspace/code/point-to-pose-model-based`, branch
`articulated/split-and-joint-fixes`. Do not touch user-owned untracked LightGlue/,
tapnet/, segment-anything-2-real-time/, inspect, or test/reconstruction/.

## Working environment

Use `/home/smkim/miniconda3/envs/point2pose_model/bin/python`.
GPU commands need host/GPU permission. Environment:

```bash
export PATH=/home/smkim/miniconda3/envs/point2pose_model/bin:$PATH
export CUDA_HOME=/home/smkim/miniconda3/envs/point2pose_model
export CPATH=/home/smkim/miniconda3/envs/point2pose_model/targets/x86_64-linux/include:/home/smkim/miniconda3/envs/point2pose_model/include
export TORCH_CUDA_ARCH_LIST=8.9
```

Read `doc/partnet_dev_set.md`, `doc/reprojection_ablations.md`, and
`doc/part_surface_memory.md` for earlier results. Generated data and videos live
under ignored `results/`. No manipulator mask removal is being pursued.

## Established methods to prioritize

- [BundleTrack (IROS 2021)](https://arxiv.org/abs/2108.00516): memory-augmented pose
  graph optimization for model-free object tracking. Its official keyframe
  matching/selection implementation has now been inspected; a small independent
  feature/keyframe diagnostic is documented in `doc/keyframe_baseline.md`.
  Neither a complete reproduction nor a BundleTrack baseline has been run.
- [Sturm, Stachniss and Burgard (JAIR 2011)](https://arxiv.org/abs/1405.7705): noisy
  part-pose observations to kinematic graph/model selection and pose prediction.
  Audit the existing JointModel against this established formulation, particularly
  observability, model selection and reliable observation histories. This method
  assumes useful pose observations; it cannot justify feeding bad tracks into a
  confident joint fit.
- [ArticulatedFusion (ECCV 2018)](https://arxiv.org/abs/1807.07243): joint motion,
  geometry and segmentation with two-level motion optimization. Inspect its
  rigid-group/segmentation coupling before proposing a new online grouping method.
- [URDF joint specification](https://docs.ros.org/en/rolling/p/urdfdom_headers/generated/classurdf_1_1Joint.html):
  origin maps parent to joint frame; axis is in joint coordinates. The exporter
  fixes below follow this convention, not a new articulation method.

The current reprojection gate, adjacent-frame fits, temporal priors and frame
conversions are standard components or engineering adaptations. Do not claim them
as independently novel. Keep the causal opposite-face part identity problem as a
research question and record matched ablations.

## Pose diagnostics from this iteration

`replay --trace-out <file.npz>` now archives every persistent ID, including IDs
later removed, with per-frame poses, presence, observed flags, seed-label GT votes
and evaluation-only GT poses. It uses no pickle. Trace data never feeds tracking.
The ablation runner accepts `--trace`.

For `laptop_hinge_orbit`, the original p0 follows the **lid**, not the base, until
frame 41. At frame 41 its lid-aligned error is 34 mm / 1.9°. Frame 42 jumps 71.6°;
GT changes are 3.03° for the base and 7.42° for the lid. Do not interpret the
pre-split base-aligned error as a pose failure: a single discovered group can
follow either real part before articulation is resolved.

Diagnostic outputs: `results/pose_diagnostics/{baseline,continuity,bounded,
incremental,incremental_nogate,continuity_low_support}.{npz,mp4}`. Corresponding
logs are `/tmp/pose_diagnostic.log`, `/tmp/pose_continuity.log`,
`/tmp/pose_bounded.log`, `/tmp/pose_incremental.log`,
`/tmp/pose_incremental_nogate.log`, `/tmp/pose_low_support.log`.

- Temporal continuity with 30% surface support fails to rescue frame 42.
- Bounding dense-refinement corrections changes nothing; no large refinement
  correction was observed. The refiner was therefore not responsible for this jump.
- Unconditional adjacent-frame rigid fitting reduces the frame-42 jump to 4.62°
  but accumulates drift and loses lid motion. Disabling the split gate produces
  early false prismatic joints and a 105.53° jump. Neither is a good default.
- Relaxing temporal support to 10% reduces the frame-42 jump to 7.39° and keeps
  lid error at 49 mm / 2.8° there, but loses the lid later. A threshold relaxation
  delays failure; it does not recover missing reverse-face geometry.
- **Guarded adjacent-frame fitting** uses short-baseline motion only when the
  anchored sparse pose proposes a >20° rotation change. This avoids unconditional
  drift and yields 2/2 GT coverage in the combined clip, 90.8% purity, with a late
  lid ID tracked for 41 frames at 36.5 mm / 4.65° aligned median error. Split is late
  at frame 80. Main-ID full-history base error remains 655.8 mm / 73.35° and mixes
  the pre-split lid-following phase with the eventual base identity. Do not hide it.

The measured controls are in `results/pose_controls_v1/summary.json` (unconditional)
and `results/pose_guarded_v1/summary.json` (guarded, four clips):

| Variant | Locked orbit | Clean hinge | Drawer |
|---|---|---|---|
| Reprojection split reference | 1 part, 22.0 mm main-ID error | 2 parts, 96.6% purity; final lid ID only 11 frames | 2 parts; drawer 5.1 mm / 0.67° |
| Unconditional incremental | 2 parts, 63.0 mm main-ID error | 2 parts, 88.1%; lid 430.0 mm over 56 frames | drawer 11.3 mm / 1.19° |
| Guarded incremental | Identical to reference | 2 parts, 88.8%; lid 434.2 mm over 47 frames | Identical to reference |

The clean-hinge pose windows differ and cannot support a fair direct accuracy
claim. Keep all new pose presets experimental. No default was changed. Remaining
controls for guarded mode: static and noisy hinge; rerun RBO only after simulation
metrics justify it. Adjacent-frame fitting preserves the existing NumPy RNG state
so adding a hypothesis does not change subsequent split sampling through RNG use.

## URDF conversion now validated

Fixed two separate problems: exporter omitted the fitted rest transform and used
parent-frame axes as joint-frame axes; viewer ignored origin rotation/translation
and used an incorrect pivot formula. It also lost joint-to-q indexing when walking
an out-of-order tree. The exporter now preserves the fitted root/tree and rejects
unfitted/disconnected/cyclic graphs instead of silently writing an invalid URDF.
It no longer re-roots by size with only axis negation.

For a hinge, `A(q)=A0 T(c) R(axis_local,q) T(-c)`. A joint-frame helper link followed
by a fixed mesh-frame offset preserves each physical part's canonical coordinates.
Prismatic and fixed models use their full rest transform too. The viewer composes
standard URDF origins and motions and hides fixed helper joints from motion sweeps.
Eight unit tests cover nontrivial rest rotation/translation, off-origin oblique
hinges, sliders, fixed joints, nonzero roots, chains and invalid graph rejection.

An actual learned drawer artifact was produced from RGB-D + union mask:

- `results/urdf_validation/drawer.urdf`
- `results/urdf_validation/meshes/part0.obj`, `part1.obj`
- `results/urdf_validation/drawer_gaussians.npz`, `drawer_joints.npz`
- `results/urdf_validation/drawer_tracking.mp4`, `drawer_animation.mp4`
- `results/urdf_validation/kinematics_check.json`

The learned slider spans 109.4 mm (GT scripted travel about 109.6 mm). Its tracking
run retains 97.6% sparse purity, 5.1 mm / 0.67° drawer-aligned error and 4.7° axis
error. SAPIEN loads 3 links (2 physical + 1 fixed adapter) and 1 active prismatic
joint. Across 11 joint states, the maximum elementwise matrix discrepancy against
standard URDF FK is 3.47e-7. This verifies loading/kinematic consistency, not GT
geometry accuracy. The Poisson meshes are partial, imperfect observed surfaces;
inertial parameters remain placeholders, not identified physical properties.

Reproduce kinematic validation:

```bash
python scripts/sim/check_urdf_kinematics.py results/urdf_validation/drawer.urdf \
  --out results/urdf_validation/kinematics_check.json
python -m examples.multi_part.urdf_view results/urdf_validation/drawer.urdf \
  --out results/urdf_validation/drawer_animation.mp4 --frames 48
```

## Next concrete work

1. The first RGB-D keyframe control is complete (`doc/keyframe_baseline.md`).
   LightGlue previous-only / best-keyframe / pooled-keyframe median errors are
   145.0 / 128.6 / 40.3 mm on the locked orbit. All observe 120 frames, but none
   establishes a replacement for the existing approximately 22 mm tracker.
   Next inspect robust joint keyframe refinement and reprojection consistency;
   pooled fixed poses are not bundle adjustment. Test additional views/objects
   before coupling articulation. Implementation and runner are
   `examples/multi_part/keyframe_rigid.py`, `scripts/sim/keyframe_baseline.py`.
2. Shared-frame evaluation is now implemented in
   `scripts/sim/compare_pose_traces.py`; see `doc/common_window_evaluation.md`.
   Combined-motion guarded lid has only 31/119 qualifying observed frames,
   versus zero for reference, so comparative lid accuracy is unavailable.
   Clean-hinge incremental and guarded both exceed 430 mm on 44 shared frames.
   Do not mistake held poses or late discovery for continuous recovery. Keep
   coverage and alignment limitations alongside every accuracy comparison.
3. Couple part discovery with trustworthy pose histories and inspect existing
   Sturm-style model selection. Track type/axis confidence over time, not just the
   final or earliest confident label. Avoid false early "controllable" claims.
   Axis bootstrap confidence audit found and fixed a dispersion bug and
   missing-evidence overconfidence (`doc/joint_confidence_audit.md`). Saved
   drawer pose prefixes preserve all 27 type decisions. Full guarded drawer and
   clean-hinge reruns at `4a7eca3` have exactly identical saved poses/observed flags
   to the earlier controls (`results/confidence_controls_v1`). Hinge still fails.
   Readiness logging now requires valid/excited articulation and avoids calling
   an unverified estimate controllable. Remaining controls: combined motion,
   noisy hinge and new objects; prioritize actual tracking improvements next.
4. Validate the corrected URDF exporter on a reliably learned hinge as well as the
   slider, then load and render the artifact in SAPIEN. GT/oracle joint/pose tests
   are useful diagnostics but must not be presented as online inference results.
   Latest matched joint-tracking ablation (`doc/joint_tracking_ablation.md`):
   removing the scalar override reduces shared hinge error 432 to 119 mm but
   loses six observed frames and introduces an ID switch. Experimental
   `joint_reprojection_gate` reproduces no-joint hinge and mostly retains drawer
   overrides, with no evidence to promote it. Defaults unchanged. Missing-surface
   pose initialization/identity remains the next substantial tracking target.
   Recovery eligibility ablation now recorded (`doc/surface_recovery_retry.md`):
   retrying a depth-contradicted observed pose adds one FALSE recovery in combined
   motion (frame 110, 75 -> 445 mm). Keep `recovery_contradicted_pose` off. Fresh
   traces contain separate `recovered` flags; diagnostic branch counters explain
   skipped searches. Next inspect other-part surface confusion / canonical model
   bias before relaxing gates. Clean hinge remains unchanged.
5. Keep matched ablations and negative results. Commit/push tested checkpoints,
   leave larger data local, and report meaningful progress in Korean.

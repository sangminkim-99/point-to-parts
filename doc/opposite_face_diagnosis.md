# Opposite-face relock: measured failure and next baseline comparison

The tracking worker reproduced the rigid thin-slab turnover failure with an
epoch-attributed trace: `results/thin_turnover_diag_v1/rigid_relock.json`, report
`results/thin_turnover_diag_v1/turnover_relock_diagnosis.md`. This is a scripted
simulation with oracle binary union masks; GT is used only for evaluation.

The first wrong relock is at frame 47. All eight fitting tracks are pre-flip
anchors, not newly created points. Anchor count stays at 224 while the pose is
held at frames 30–46. The reseed residual guard prevents additions during that
hold. This contradicts the proposed explanation that held-pose anchoring caused
this particular relock.

Old correspondences reappear on the opposite face and support a low-residual
but incorrect rigid transform. Later, the observed-pose error approaches 180
degrees while residual falls below a millimetre. Newly reseeded tracks mostly
disagree with this transform, rather than sustain it. There is also pre-hold
error: about 20 degrees at frame 25 and 47 degrees at frame 29 while support
shrinks. These numbers describe this synthetic sequence, not real-camera
accuracy or a universal mechanism.

## Existing methods to compare before adding another mechanism

Use identical RGB-D input, oracle union mask, random seed and tracker checkout:

| Config | Existing method |
| --- | --- |
| reprojection_split.yaml | anchor-based sparse rigid fitting |
| incremental_pose.yaml | compose adjacent-frame rigid motion |
| guarded_incremental.yaml | use incremental fit for large anchor-pose jumps |
| pose_continuity.yaml | score predicted/refined geometry with continuity preference |

Report rotation/translation error conditioned on observed vs held, incorrect
observed-pose rate, recovery latency after turnover, false splits, and runtime.
A pose that merely remains held forever is not successful recovery.

`pose_continuity` is NOT a hard rejection rule. In `choose_pose`, if neither
temporal candidate has sufficient support, the function returns the original
sparse estimate even if discontinuous. The existing unit test
`test_continuity_never_rescues_an_unsupported_prediction` covers that behavior.
Therefore enabling the option alone does not guarantee honest uncertainty after
a lost view. Any later change to reject all candidates must propagate a held
status and prevent map/joint updates; it should not silently label a prediction
as a fresh observation.

This matched ablation is assigned to the existing Fable tracking worker.
No defaults have changed on the strength of this diagnosis.

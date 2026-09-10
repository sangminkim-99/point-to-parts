# Hand-free PartNet-Mobility development set

Small diagnostic suite for articulated tracking, generated with locally installed
SAPIEN 3.0.3 and PartNet-Mobility assets. There are no hands or manipulators.
This is a development set, not a held-out benchmark or a substitute for RBO.

## Cases

Each clip has 120 frames at 640×480, nominally 30 Hz (4 seconds).

| Clip | Asset | Motion | Observable GT parts |
|---|---|---|---:|
| laptop_static | 9748 | Locked laptop, fixed camera | 1 |
| laptop_hinge | 9748 | Open/close, fixed camera | 2 |
| laptop_orbit | 9748 | Locked laptop, 360° camera orbit and return | 1 |
| laptop_hinge_orbit | 9748 | Open/close with 360° orbit | 2 |
| laptop_hinge_noisy | 9748 | Same hinge motion, depth noise σ=0.003 z² m | 2 |
| storage_slide | 40147 | Drawer only; door locked | 2 |

The hinge traverses 132°, and the drawer travels 110 mm. Motion clips hold still
at both ends and reach the maximum scripted state before returning. A full orbit
in four seconds is an intentionally demanding visibility/identity test.
All depth PNGs use integer millimetres, including the nominally clean cases.
Noise uses a fixed seed; at 0.7 m its standard deviation is about 1.47 mm.

## Generate and validate

Activate an environment with SAPIEN 3, NumPy and OpenCV (locally
`/home/smkim/miniconda3/envs/point2pose_model`). Offscreen rendering needs GPU access.

```bash
python scripts/sim/build_dev_set.py \
  --assets /home/smkim/workspace/dataset/partnet_mobility \
  --out results/partnet_dev_v1
```

The versioned recipe is `configs/sim/partnet_dev_v1.json`. Existing matching cases
are validated and reused; changed settings require a new output directory.
Add `--validate-only` to inspect an existing suite without rendering.
Large RGB-D files and previews stay in the ignored results directory; source
assets are referenced locally, not copied into Git.

Each case contains RGB, depth, modal part masks, intrinsics, camera/part poses,
source joint states, timestamps, link grouping, a render command, and a six-frame
`preview.jpg`. `manifest.json` summarizes validation across the suite.
Joint state is set directly without advancing physics, so gravity cannot move
links between the scripted state and GT capture.

GT parts are *observable rigid motion groups*: fixed joints and joints that never
move in the clip are merged. For example a locked laptop is one rigid part even
though its URDF has a hinge. Source joint states and original link membership are
retained so that this choice is auditable. These GT poses are simulator link-frame
poses; compare tracker trajectories with one fixed gauge alignment per part.

Validation checks frame counts, labels/depth validity, rigid transforms, visible
GT groups, and absence of image-boundary clipping. It also projects the renderer's
camera-space position buffer into image pixels before adding noise. The observed
maximum error is under 0.047 pixels across all six clips. Pixel centers here use
the renderer's `(u+0.5, v+0.5)` convention. This check validates image/depth
alignment, not tracker accuracy or temporal correspondence.

## Tracker baseline and reprojection work

```bash
python -u -m examples.multi_part.replay \
  --seq-dir results/partnet_dev_v1/laptop_hinge \
  --method naive --config surface_memory.yaml \
  --stride 1 --view 0 --hyp-panel 0 \
  --out results/partnet_dev_v1/laptop_hinge/tracking.mp4
```

Use the CUDA environment setup in `doc/rbo_sam2_demo.md` if needed. The tracker
receives RGB-D and the **union object mask**; individual GT part labels/poses are
reserved for evaluation. Starting with an exact object mask isolates tracking
from segmentation errors. SAM2 can subsequently replace that union mask for a
separate segmentation experiment.

For the opposite-face problem, use the orbit cases to test whether an existing
part ID survives disappearing texture and reappearance. Reproject stored
part-local geometry using the estimated camera/part transform, apply depth-based
visibility checks, and compare predicted silhouette/depth with current RGB-D.
An unseen reverse face should extend the existing part model when supported by
rigid/joint-consistent motion. Camera motion alone must not create a second part.
The known poses permit an oracle reprojection experiment to separate geometry
coverage failures from pose-estimation failures. These oracle poses must stay out
of the normal tracker input. Gaussian rendering can add color/silhouette losses
once this geometric association is reliable.

## Initial baseline (2026-09-10)

Run with `--method naive --config surface_memory.yaml --stride 1` at generator
commit `e5b6669`. These are single-run diagnostics, not aggregate benchmark scores.

| Case | Final parts / GT | Sparse label purity | Diagnosis |
|---|---:|---:|---|
| Static | 1 / 1 | 100.0% | No false split; aligned pose median 1.2 mm / 0.12° |
| Hinge | 2 / 2 | 96.6% | Extra split at frame 110; final lid ID has only 11 frames of history |
| Orbit only | 3 / 1 | 100.0% | False articulation caused by changing visibility |
| Hinge + orbit | 2 / 2 | 68.9% | Pose failure; 79% of current object surface unmodelled |
| Hinge + noise | 3 / 2 | 78.5% | Over-segmentation and severe lid pose drift |
| Drawer | 2 / 2 | 97.6% | Correct prismatic type; drawer aligned median 5.1 mm / 0.67°, axis 4.7° |

Purity is a sparse final-point label score and does not penalize splitting one GT
part into several predicted parts: orbit-only gets 100% while clearly failing.
Always read it alongside part count, persistent-ID history, and aligned trajectory
errors. The hinge's final lid has a median aligned error of 90.0 mm / 8.16° over
only 11 frames, so its high purity does not mean successful tracking. Hinge+orbit
has approximately 1 m lid trajectory error and an incorrect prismatic joint.
Depth-only recovery triggered zero times on all six baselines; these results do
not validate that fallback on real rendered view changes.

Local `tracking.log` and `tracking.mp4` in every case retain the full outputs.
No tracker tuning was applied to these six clips. Prioritize the locked orbit
case for reprojection-based association first, then the combined hinge/orbit case.
A useful success criterion is one continuous ID under camera-only motion, then
two continuous IDs and correct hinge motion under articulation. This avoids
accepting a final part count or purity score that hides intermediate identity loss.

## Reprojection follow-up

The experimental `--config reprojection_split.yaml` adds visibility-aware RGB-D
validation before accepting a split. It fixes the locked-orbit false split and
reduces noisy-hinge over-segmentation, while combined hinge/orbit still fails and
loses GT coverage. Keep the original preset for baseline comparisons. See
`doc/reprojection_ablations.md` for the full 24-run matrix, matched sparse/dense
controls, temporal-pose negative results and research claim boundaries.

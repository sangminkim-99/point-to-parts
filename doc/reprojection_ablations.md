# Reprojection experiments: claims and ablations

## Question and implemented mechanism

Can an online tracker distinguish a newly visible face from a newly articulated
part? Sparse track consistency alone fails when old tracks migrate to the reverse
face. The first intervention validates a proposed split against stored canonical
geometry and the current RGB-D observation, before allocating a new persistent ID.

For each proposed group, canonical Gaussian centers are assigned by nearest
canonical tracked point, then projected under the common rigid pose and the
proposed separate pose. Scores use a fixed sample denominator. Depth-consistent
object pixels support a pose; predicted geometry in front of measured depth or
in valid background contradicts it. Geometry behind measured depth is neutral.
A split requires sufficient support for both groups and a reduction in observable
contradictions in at least one group. Missing/occluded geometry is not positive
articulation evidence. No GT camera poses, part masks, joint states or URDF are
used by this decision. Input segmentation remains the single union object mask.

This is conservative model selection, not a complete solution to pose tracking.
Tangential motion with little depth/silhouette change may be delayed. Missing
surface support can also defer a real split. The known hinge test still loses its
lid ID; final part count and purity alone are insufficient success criteria.

## Experiment protocol

`scripts/sim/ablate_reprojection.py` stores the command, Git commit, working diff,
dataset metadata, full log, video and parsed metrics for each case/variant.
Results are incremental in `summary.json`; an existing run is not overwritten.
Use the environment/CUDA setup in `doc/rbo_sam2_demo.md`:

```bash
python scripts/sim/ablate_reprojection.py \
  --data results/partnet_dev_v1 --out results/partnet_ablation_v1
```

The primary controls are locked orbit (must never split), hinge (must discover
motion), and drawer (different joint type). Static, noisy hinge and combined
articulation/orbit are additional stress checks. These assets have already been
used for development, so none of these results are held-out generalization.
Runs are single deterministic-style replays (the existing registration backend
resets its NumPy seed); hardware-level reproducibility is not guaranteed.

| Factor | Comparison | Claim it can test |
|---|---|---|
| RGB-D verification | Sparse split decision vs extra geometry check | Fewer visibility-induced false splits without suppressing true motion |
| Occlusion treatment | Behind-surface neutral vs symmetric depth error | Visibility, not simply adding a residual, prevents false articulation |
| Surface memory | Gaussian centers vs tracked canonical points | Persistent surface coverage adds evidence after feature loss |
| Pose hypothesis source | Sparse fit vs temporal geometric alternative | Identity stability can be accompanied by lower pose drift |
| Joint recovery | Free motion vs known-joint depth search | A previously learned joint recovers an existing part after feature loss |
| Appearance representation | Geometry only vs Gaussian RGB/silhouette optimization | Gaussian appearance contributes beyond depth alignment |

Only experiments actually run below support a claim. The last two rows are
planned: current depth-only joint recovery has not triggered on these clips and
Gaussian color optimization has not been compared. Existing Gaussian centers are
used as geometry here, which does not establish a Gaussian-specific advantage.

## Novelty boundary

Reprojection, reconstruction/tracking coupling, and occlusion handling individually
are established ideas. [ArticulatedFusion (ECCV 2018)](https://openaccess.thecvf.com/content_ECCV_2018/html/Chao_Li_ArticulatedFusion_Real-time_Reconstruction_ECCV_2018_paper.html)
already addresses simultaneous motion, geometry and segmentation reconstruction.
[BundleSDF (CVPR 2023)](https://bundlesdf.github.io/) couples object tracking with
online reconstruction. [OcclusionFusion (CVPR 2022)](https://arxiv.org/abs/2203.07977)
explicitly studies occlusion-aware motion for dynamic reconstruction.

A possible contribution to investigate is **causal, visibility-conditioned part
birth and identity retention across opposite faces**, with online joint discovery
and explicit evidence accounting. That is a research hypothesis, not a verified
novelty claim. Evidence needs multiple assets/categories, camera speeds, joint
motions and occlusion durations; part-birth delay, ID switches, pose/joint errors,
visible-surface coverage and runtime; and meaningful comparisons to prior methods.
Ablations explain a method's contribution but do not establish novelty by themselves.

## First measured results

At `90af7cc` / the recorded working patch, the original split gate changes the
locked orbit from 3 predicted parts to 1, with no accepted split. Hinge and drawer
retain exactly their baseline split histories and final metrics. The orbit's
aligned translation error changes from 20.1 mm (the baseline's main fragment) to
22.0 mm for the retained whole-object ID. This is not a pose-accuracy improvement;
the tracked geometry/membership also differs between those two trajectories.

The symmetric-depth control returns to 3 parts on locked orbit, while retaining
hinge/drawer results. This supports the specific visibility interpretation in this
one sequence, not a general performance or novelty claim. Sparse-only with the
original minimum of 20 geometry points suppresses the hinge entirely (1/2 parts,
52.0% purity). Its apparent success on orbit is confounded by rejecting under-sized
geometry groups. A matched 6-point-minimum dense/sparse comparison is therefore
run separately, with both overrides explicit and archived.

The temporal pose option extrapolates the previous two estimated poses, refines
the prediction against self-visible stored geometry, and lets it compete with the
sparse fit using fixed-sample depth support minus free-space contradictions. It is
independent of the split gate and off by default. It must improve trajectory error
and ID history before becoming a recommended setting; merely selecting a temporal
hypothesis is not evidence that it was correct.

## Matched primary results

Each cell is final predicted parts / sparse purity; GT part counts are in the
column headers. Full pose histories, split events and commands are in the local
`results/partnet_ablation_v1` and `results/partnet_ablation_v2` artifacts.

| Variant | Orbit (GT 1) | Hinge (GT 2) | Drawer (GT 2) |
|---|---:|---:|---:|
| Baseline | 3 / 100.0% | 2 / 96.6% | 2 / 97.6% |
| Dense + occlusion-aware | 1 / 100.0% | 2 / 96.6% | 2 / 97.6% |
| Dense + symmetric depth | 3 / 100.0% | 2 / 96.6% | 2 / 97.6% |
| Dense + occlusion-aware, min 6 | 1 / 100.0% | 2 / 96.6% | 2 / 97.6% |
| Sparse + occlusion-aware, min 6 | 2 / 100.0% | 1 / 52.0% | 2 / 97.6% |
| Dense + temporal pose option | 1 / 100.0% | 2 / 88.8% | 2 / 97.6% |

The matched minimum-6 comparison removes the sparse sample-count confound: dense
geometry keeps one orbit part and discovers the hinge; sparse geometry creates
two orbit parts and misses the hinge. The drawer remains unchanged. This supports
the value of persistent surface samples on these cases, not Gaussian appearance
or differentiable rendering specifically. Dense samples are not independent
measurements: they were reconstructed using earlier estimated poses.

Temporal pose selection fires five times on the hinge. It keeps lid ID 1 from
frame 74 to the end (47 evaluated frames), avoiding the additional split, but has
274.4 mm / 28.27° median aligned lid error and lower purity. Baseline's final lid
ID 2 exists for only 11 frames, with 90.0 mm / 8.16° error. Those different time
windows cannot establish a fair numerical pose improvement or degradation. The
absolute errors and purity do establish that this is not solved. Keep the option
off by default. It never fires on orbit or drawer and changes neither result.

Sparse-min6 orbit invokes joint-depth recovery for 9 frames after inventing an
incorrect extra part. This is a failure, not validation of useful recovery.

The paired orbit median step timings were 108.7 ms baseline and 108.9 ms proposed
in this single run (both about 9.2 FPS). Timing excludes input/mask generation,
evaluation and video writing. This is not an end-to-end real-time claim.
`results/partnet_ablation_v1/orbit_comparison.mp4` presents synchronized baseline
and proposed outputs at 15 FPS playback; the source sequence is nominally 30 Hz.

## Six-case regression and current recommendation

The three additional proposed runs complete coverage of the original six clips:

| Case | Baseline | Proposed | Interpretation |
|---|---|---|---|
| Static | 1 part, 100% purity | Same, aligned 1.2 mm / 0.12° | Preserves the negative control |
| Noisy hinge | 3 parts, 78.5% purity | 2 parts, 87.4% purity | Fewer false parts; lid still drifts badly (359.2 mm / 35.56° over 47 frames) |
| Hinge + orbit | 2 parts, 2/2 GT covered, 68.9% purity | 2 parts, **1/2 GT covered**, 91.9% purity | Regression in coverage: both predicted parts map to base; lid is missed |

Higher purity in the combined case is misleading and is **not** an improvement.
The main predicted pose has approximately 1.06 m / 158° aligned error. The split
gate cannot fix a common-pose hypothesis that has already failed. Keep
`reprojection_split.yaml` as an experimental preset; do not replace the general
`surface_memory.yaml` default based on these six development sequences.

There are 24 completed matrix/regression runs (12 original, 9 matched-minimum and
pose-option, 3 additional stress runs), plus three initial pilot gate runs. No
held-out generalization has been measured. The first matrix spans small changes
to diagnostic logging only; numerical options used there keep the original
20-point minimum and temporal pose disabled. Per-run commits/patches record this.
Later matrices use `17f3aa4` with unchanged tracking code.

Next technical priority is recovery/selection of a valid common pose under
simultaneous camera and joint motion, with fair pose-error evaluation over the
same GT frames across ID births/deaths. Validate that before interpreting joint
axis estimates, URDF output, or adding an appearance optimizer. The temporal
candidate pilot above is insufficient. An oracle-camera/pose diagnostic could
separate association failure from pose failure, but must be labelled explicitly
and never mixed into the causal tracking results.

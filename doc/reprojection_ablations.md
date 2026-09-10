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

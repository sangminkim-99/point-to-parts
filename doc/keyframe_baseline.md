# RGB-D keyframe tracking controls

2026-09-10. This is a rigid tracking diagnostic, not an articulated reconstruction
result or a reproduction of BundleTrack. No tracker default changed.

## Established components

Read the official [BundleTrack implementation](https://github.com/wenbowen123/BundleTrack)
at commit `3d0544a9fa2b05dd75a61e96fecfdf27ca2fab68`, especially `src/Bundler.cpp`:
previous-frame feature registration, keyframe retention, and local bundle
adjustment selection. Its full LF-Net / C++ CUDA optimization stack was not run.
The experiment below isolates feature registration and memory with the existing
environment. It does not implement BundleTrack's joint keyframe optimization.

Features are either OpenCV SIFT with RootSIFT normalization and mutual ratio-test
matching, or SuperPoint (1024 keypoints) plus the official pretrained
[LightGlue matcher](https://github.com/cvg/LightGlue). The installed LightGlue
revision is `eb42fee2d71449efb0aa5c10549752b5d75384d8`. Its user-owned checkout was
not modified. Learned weights were cached in ignored results storage.

Valid object RGB-D features are lifted to 3D. Three-point SE(3) RANSAC uses an
8 mm residual threshold, 128 trials, and at least eight inliers. A successful
previous-frame match is composed into the initial-camera gauge. Memory adds a
keyframe every ten frames, holds at most twelve, and preserves the first view.
Single-reference selection maximizes inlier count, then minimizes residual.

The optional multi-reference variant takes at most 32 inlier pairs from each of
the four highest-ranked references, transforms them into the original gauge,
and fits one current pose using the pooled pairs. Stored poses remain fixed:
this is not pose-graph optimization or bundle adjustment. Failure holds the last
pose, marks it unobserved, and does not insert a keyframe. All inputs are causal.

## Measured controls

All runs use the same 120-frame `partnet_dev_v1/laptop_orbit` locked laptop clip,
RGB-D and the exact union object mask. Part/joint poses are read only after tracking
for evaluation. The script rejects clips with multiple GT motion groups: this is
an explicitly restricted rigid control, not automatic articulation discovery.
Errors are relative to frame zero. Medians include held failure frames.

| Features | Pose references | Observed / 120 | Median mm | Median degrees | Final mm / degrees |
|---|---|---:|---:|---:|---:|
| RootSIFT | Previous only | 82 | 92.7 | 8.67 | 982.5 / 113.19 |
| RootSIFT | Keyframes, best one | 86 | 67.9 | 6.59 | approximately 0 / 0 |
| SuperPoint + LightGlue | Previous only | 120 | 145.0 | 12.33 | 280.8 / 25.36 |
| SuperPoint + LightGlue | Keyframes, best one | 120 | 128.6 | 10.95 | approximately 0 / 0 |
| SuperPoint + LightGlue | Keyframes, pooled | 120 | 40.3 | 3.47 | 4.0 / 0.31 |

RootSIFT on the static control observes all 120 frames with near-zero error.
Outputs are `results/keyframe_baseline_v1/{orbit_previous,orbit_memory,static_sift,
orbit_lightglue_previous,orbit_lightglue,orbit_lightglue_multi}/`: each has
`summary.json` and `trace.npz`. The last two newly launched runs save the complete
command and implementation hash; earlier summaries predate that metadata.

The first and last orbit render are identical. Near-zero endpoint error from
retaining the first keyframe demonstrates revisit recovery on this synthetic
sequence, not accurate intermediate tracking or general reverse-face recognition.
All-frame errors show substantial drift despite successful feature fits.
The existing reprojection-split tracker reports about 22 mm / 1.8 degrees on
119 frames aligned from frame one; this slightly different window precludes an
exact matched ranking, but there is no evidence to replace it with this baseline.
Runs shared CPU/GPU resources, so recorded latency is not a controlled speed
comparison and does not establish real-time performance.

## Interpretation and next checks

Memory helps recovery; pooling multiple references improves this LightGlue run
over selecting one reference. A higher match/observation count alone does not
mean more accurate tracking. Neither pooling nor memory is claimed as novel.

Before integration, test robust keyframe pose refinement with fixed first-frame
gauge and reprojection/depth consistency, then evaluate on additional locked
objects and partial revisits. Stored poses and false correspondences can both
bias the current fit. This experiment does not connect different unseen lid
surfaces by itself. Articulation coupling still needs reliable shared motion
histories and explicit joint-model selection.

## Reproduce

Use the point2pose_model Python environment. RootSIFT only needs OpenCV/NumPy;
the learned backend additionally needs the installed LightGlue package, PyTorch,
CUDA and its official pretrained SuperPoint/LightGlue weights.

```bash
export TORCH_HOME="$PWD/results/keyframe_baseline_v1/torch"
python scripts/sim/keyframe_baseline.py \
  --seq-dir results/partnet_dev_v1/laptop_orbit \
  --out results/keyframe_baseline_v2/orbit_lightglue_multi \
  --backend lightglue --multi-reference
```

Drop `--multi-reference` for best-reference memory; use `--previous-only` for
the incremental control. Use `--backend sift` for the classical feature control.
Choose a new output directory for each run. Five unit tests cover robust fitting,
degenerate geometry, proper planar rotation, failed-frame memory protection, and
noncommuting pose composition across multiple stored camera frames. The relevant
tracking/export/simulation test suite passes 48 tests.

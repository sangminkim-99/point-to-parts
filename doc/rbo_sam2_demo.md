# RBO trial with causal SAM2 object masks

Tested 2026-09-09 on an RTX 4070 Ti using the `point2pose_model` conda environment.
SAM2.1 Hiera Large receives one whole-object bounding box derived from GT on the
first usable frame. Usable-frame selection also uses GT. After initialization,
SAM2 propagates causally; no per-part GT masks are passed to the tracker.
GT is used separately for evaluation. These are exploratory single runs, not a
controlled comparison against the previous masking method.

| Sequence / setting | Source frames | Initial points / TAPIR iterations | Parts found / GT | Median step FPS | Result |
|---|---|---|---|---|---|
| cardboardbox01_o, dense_refine | 0–239 of 409 | 200 / 1 | 1 / 2 | 15.5 | No split or joint |
| cardboardbox01_o, dense_refine, max_points=600 | 0–239 of 409 | 300 / 4 | 1 / 2 | 5.3 | No split or joint |
| ikeasmall02_o, dense_urdf | 2–241 of 265 | 200 / 1 | 4 / 3 | 12.5 | Covers all GT parts, but over-segments and fits an incorrect revolute joint |

Timing is `tracker.step`, including dense reconstruction/refinement, but excludes
SAM2, image loading, GT evaluation, visualization and video encoding. It is not
end-to-end stream FPS. P90 step latency was 73.2, 200.0 and 165.5 ms respectively.
The drawer preset enables retrospective fitting from already observed tracks
and allows 600 tracks; the box default allows 400. Refinement is projective depth
ICP, not differentiable RGB rendering optimization.

## What the outputs show

- Box: SAM2 follows the opening lid and excludes most of the hands in inspected
  frames. Lid tracks disappear under hand occlusion, however, and persistent
  motion evidence is insufficient to split. Increasing TAPIR effort does not
  resolve this. The model grows newly visible lid geometry into the same part.
- Drawers: initial split at internal frame 24; subsequent splits at 167 and 193.
  Sparse-label purity is 94.8%, with coverage of all three GT parts. The evaluator
  reports a 5.3-degree axis error for one matched prismatic joint only; this is
  not an accuracy score for the whole kinematic model. Final fitted joints are
  prismatic, revolute, prismatic, whereas the object has two sliding drawers.
  The extra part has only 83 active Gaussians. Reported confidence is therefore
  insufficient by itself to certify the discovered model.
- Drawer SAM2 masks include the manipulation stick once it enters. A negative
  prompt during propagation or a separate manipulator mask is needed to exclude
  it; the high aggregate IoU hides this contamination. Its contribution to the
  extra part has not been isolated.
- Whole-object union IoU averages 0.653 for the box and 0.928 for the drawers.
  RBO GT masks are mesh projections that do not exclude external occluders such
  as hands. These values must not be described as visible-object mask accuracy.

## Reproduce

Activate the existing environment so Ninja is available to gsplat. CUDA kernels
compile on first use; that startup should not be interpreted as steady-state
tracking latency.

```bash
conda activate point2pose_model
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CONDA_PREFIX/include"
export TORCH_CUDA_ARCH_LIST=8.9

seq=/home/smkim/workspace/dataset/RBO/sequences/ikeasmall02_o
out=results/rbo_sam2_demo/ikeasmall02
mkdir -p "$out"
python -u -m experiments.articulated.run_sam2_propagate \
  --seq-dir "$seq" --init bbox --stride 1 --max-frames 240 --no-show \
  --out "$out/sam2.mp4" --save-masks "$out/masks.npz"
python -u -m examples.multi_part.replay \
  --seq-dir "$seq" --method naive --config dense_urdf.yaml \
  --object-masks "$out/masks.npz" --stride 1 --max-frames 240 \
  --view 0 --hyp-panel 0 --out "$out/tracking.mp4" \
  --dump-joint "$out/joints.npz" --save-model "$out/gaussians.npz" \
  > "$out/tracking.log" 2>&1
```

For the box, use `cardboardbox01_o`, a separate output directory and
`--config dense_refine.yaml`. The quality variant additionally uses
`--n-points 300 --pips-iter 4 --set max_points=600`.

Local videos, masks, logs and model snapshots live in the ignored
`results/rbo_sam2_demo/` directory. Snapshots contain means, colors, scales,
quaternions, opacity, part labels, final part-to-camera poses, parent indices,
intrinsics and source-frame ID. Label -1 denotes inactive geometry. They are
inspection snapshots, not complete resumable tracker checkpoints or URDFs.

## Next demo milestone

Add causal negative-prompt corrections to SAM2, then repeat the drawer trial
with the manipulation stick excluded. Address persistent track loss through
occlusion and prevent geometry growth from hiding unresolved part motion.
Validate discovered topology and URDF frame conventions before presenting an
exported model as usable. No URDF was exported in this trial.

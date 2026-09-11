# Viser export viewer

Save a fitted model using Save in the author demo. The directory contains
`object.urdf`, `model.npz`, and `metadata.json`. Run from the repository root:

```bash
~/miniconda3/envs/point2pose_model/bin/python -m examples.multi_part.urdf_viser \
  results/author_demo_live/<saved-directory> --port 8091
```

Open http://localhost:8091. The left dock has joint sliders in degrees/mm,
link-frame axes, and a Gaussian/point-cloud toggle. Sliders use observed limits;
these are not certified mechanical limits. URDF forward kinematics drives the
whole chain, including fixed helper frames. The saved root pose places the
object in capture-camera coordinates. The initial joint values are zero clipped
to the observed range, not necessarily the captured configuration.

New saves preserve scales, wxyz orientations, and opacity alongside positions,
colors, and part ownership. Restart the author demo before saving to use the
updated exporter. Older exports load as point clouds; missing Gaussian
parameters are not fabricated. Unassigned geometry is counted but hidden since
it has no link transform. An export without a fitted tree has no object.urdf.

Rendering uses Viser's native experimental browser Gaussian renderer; it does
not invoke CUDA or gsplat. Its appearance/sorting may differ from the pipeline's
rendered diagnostic images. This viewer targets matching author-demo exports
(part0, part1, ...), not arbitrary robot mesh URDFs. No meshes are required.

## Independent loader check

`check_urdf_kinematics.py` now sweeps each joint independently and all joints
together, plus the zero-clipped rest configuration. On saved live export
`20260910-165808-535626`, SAPIEN and the viewer's matrix composition agree over
34 configurations (5 links, 2 active joints), maximum matrix-entry difference
3.73e-7. Artifact: `results/urdf_viewer_validation/saved_box_independent.json`.
This checks exported kinematic semantics, not whether the inferred three-part
model matches the physical object or whether its joint limits are mechanical.

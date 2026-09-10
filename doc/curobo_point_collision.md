# Mesh-free point clouds and cuRobo robot spheres

The user chose point clouds instead of required object meshes and approved
cuRobo. The internal object model should keep part-local observed clouds,
persistent IDs, estimated part transforms and kinematic joints. Sparse keyframes
remain tracking memory; Gaussian appearance is optional. URDF can describe the
kinematic tree; cloud collision data is an external companion, not a standard
URDF point-cloud collision element. Existing mesh exports are preserved.

## Implemented and actually executed

`examples/manipulation/point_sphere.py` consumes cuRobo CudaRobotModel FK spheres
and returns per-sphere/per-part clearance. Centers are transformed into each
part's local frame, then queried against chunked points. Distance is nearest
point distance minus sphere radius minus configured padding. Positive is outside
the sampled inflated surface; negative is overlap. This sign differs from some
cuRobo penetration APIs. Disabled cuRobo radii <=0 are ignored. Part IDs remain
explicit through the caller's part list. Geometry is validated for finite values,
shared device/dtype and rigid transforms. Torch gradients propagate through
sphere centers and cuRobo FK. The query is piecewise differentiable and currently
brute-force/chunked, not a spatial index or custom CUDA collision kernel.

Official source [cuRobo v0.7.6](https://github.com/NVlabs/curobo/tree/v0.7.6),
commit `2fbffc35225398cf9d5f382804faa9de2608753b`, was downloaded to ignored
`results/dependencies/curobo`. We use its real CudaRobotModel and
`link_spheres_tensor`; cuRobo MotionGen is **not yet connected to this custom
collision cost**. Do not describe this as full cuRobo motion planning.

`scripts/sim/check_curobo_points.py` executes a Franka fixture on the GPU:
65 robot spheres, 15,667 reconstructed drawer base points, deliberately synthetic
placement at a moving robot sphere. It verifies negative clearance there,
positive clearance after moving the object 10 m away, and a finite nonzero
joint gradient (norm 0.2776). The fixture is not hand-eye calibration or a
physical manipulation test. Result: `results/curobo_points_v1/smoke.json`.
A short warm-run query measurement was around 0.4 ms before the additional rigid
input validation; this is not an end-to-end or planning performance claim.

## Local environment and reproduction

The existing environment was not upgraded. Extra Python dependencies reside in
`results/dependencies/python`: warp-lang 1.6.2, yourdfpy 0.0.56 and urdf-parser-py
0.0.4. The official cuRobo source is on PYTHONPATH and its required kinematics
extension is JIT built into `results/dependencies/torch_extensions`. This is a
source-based local integration, not a full installed cuRobo wheel.

```bash
export PYTHONPATH="$PWD/results/dependencies/python:$PWD/results/dependencies/curobo/src"
export TORCH_EXTENSIONS_DIR="$PWD/results/dependencies/torch_extensions"
export PATH=/home/smkim/miniconda3/envs/point2pose_model/bin:$PATH
export CUDA_HOME=/home/smkim/miniconda3/envs/point2pose_model
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=2
python scripts/sim/check_curobo_points.py \
  --cloud results/urdf_validation/drawer_gaussians.npz \
  --out results/curobo_points_v1/smoke.json
```

Warp reports a cuDeviceGetUuid driver entry-point initialization error. The tested
cuRobo Torch/CUDA kinematics and custom query nevertheless complete successfully.
Warp-based collision/ESDF paths have not been validated; do not assume them usable.

## Next integration requirements

- Use the user's actual robot configuration and camera-to-robot calibration.
  Franka is currently only a test fixture; the actual robot is pending user input.
- Connect the cloud cost to a trajectory objective / validate candidate plans;
  current queries include linear sphere sweeps (see below), but not robot self-collision, joint
  limits, dynamics, gripper contact exceptions or object motion along a trajectory.
- Keep missing surfaces and stale part poses explicit. Empty clouds return inf
  meaning no evidence, not certified free space. Interior occupancy is unknown;
  surface distance alone can miss a robot initialized inside an object.
- Validate sampling/padding against dense geometric ground truth and use motion
  interpolation or swept queries to avoid tunneling through thin surfaces.
- A 5 mm default padding is a configurable initial allowance, not a measured
  uncertainty bound. Avoid changing object geometry to hide tracking errors.

## Swept sampled-trajectory queries

`point_sphere.py` now provides `swept_point_sphere_clearance` for
`[..., T, S, 4]` sphere trajectories and `curobo_trajectory_clearance` for
`[B,T,D]` joint trajectories. It computes closest points on each straight
sphere-center segment to every sampled object point, in the part's rigid local
frame. Fixed-radius linear sphere motion is handled exactly against these points;
varying radius uses the larger endpoint radius conservatively. Activation changes
are rejected. Static segments, empty clouds, disabled spheres and gradients are
handled. The first endpoint query reuses validation; spatial indexing remains a
future optimization.

A regression test detects a sphere crossing a thin sampled surface even though
both endpoint clearances are positive. Further tests cover stationary centers,
varying radii and activation changes. GPU smoke uses actual Franka FK at five
joint samples, 65 spheres and 15,667 drawer points; output shape is [1,4,65,1].
Finite nonzero trajectory gradient norm is 0.2776. Output is
`results/curobo_swept_v1/smoke.json`; placement remains synthetic.

This is exact only for the **linear center segments**. It is not continuous
collision certification for nonlinear joint-space interpolation: curved FK paths
may leave those segments. The object poses are fixed during the query. Dense
joint sampling or conservative curvature bounds, time-varying articulation,
self-collision and MotionGen integration remain outstanding. It also does not
establish occupied interiors or treat unobserved surfaces as free space.
The relevant test suite now has 70 passing tests.

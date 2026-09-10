# Interactive model authoring checkpoint

Run from the repository root with the point2pose_model conda environment installed:

```sh
PORT=8090 OUT=results/author_demo_live examples/multi_part/run_author_demo.sh live --depth 1
# Replay a processed recording instead:
PORT=8090 OUT=results/author_demo_replay examples/multi_part/run_author_demo.sh replay <capture-directory>
```

The launcher supplies CUDA_HOME, both conda CUDA include directories on CPATH,
TORCH_CUDA_ARCH_LIST (8.9 by default for this machine), and MAX_JOBS=2.
POINT2POSE_ENV can override the environment path. Run from the repository root
so checkpoint and config paths resolve. A changed build environment may trigger
a gsplat compilation; do not interrupt another process building the same cache.

Use the OpenCV prompt to select the object. The browser shows the tracked model,
allows reference selection, recenters on its cloud, previews a frozen joint
configuration, and saves point clouds plus a kinematic URDF when a tree is fitted.
The reference pose is tracked during whole-object motion; the current tracker
still uses a rooted model. Pairwise joint discovery with root selection only at
export remains future work.

Live runs save processed RGB/depth/binary-mask frames for replay. Saved metadata
includes live/replay provenance and observed flags. Reset clears the preview and
records an event; it does not reset tracking. Observed joint ranges are not
verified mechanical limits, and URDF files contain no collision geometry.

Simulation preparation uses an **oracle binary object mask** derived from the
union of simulator labels. No per-part labels or GT poses enter tracking, but
this control does not evaluate live segmentation quality.

## Validation and open issue

The integrated object test suite checks snapshot independence, export, recording
mask fallback, reference identity handling, and framing. It does not establish
live reconstruction or manipulation accuracy.

The user has run the live demo and reported gray/missing surfaces after scanning
then articulating a box. Current dense splitting assigns historical points to
nearest sparse seeds; carve may later remove assigned geometry following depth
contradictions. Saved labels alone do not prove that sequence caused the loss.
The viewer hides labels=-1. A worker is replaying with temporal ownership/carve
instrumentation before changing assignment defaults. Geometry retention and
correct part association must be measured separately.

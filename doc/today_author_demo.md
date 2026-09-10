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

## Gaussian diagnostics (September 10 checkpoint)

Restart the demo to load the new UI. Enable **Render diagnostics**
while frames are arriving. **Diagnostic panel** selects rendered RGB, depth
residual or observed RGB. Rendering runs at a configurable target rate (0.5–5 FPS, default 5 FPS) and is
off by default. The 3D view remains a point cloud; the diagnostic image uses
actual Gaussian rasterization with approximate nearest-depth compositing across
parts (not global alpha composition). Timing and render-frame age are displayed.

Depth coverage uses object-mask pixels with valid depth; agreement is the
fraction of covered valid object pixels within 20 mm. This measures current-view
agreement, not full-object completeness. Residual heatmap saturates at 30 mm.
Held poses and rendering errors are explicit. Refiner presence indicates
configuration, not proof of a successful optimization on the current frame.

Manager validation: 110 object tests passed; actual GPU rasterizer smoke on
recorded frame50 with 2682 depth-initialized Gaussians produced RGB/depth output.
Artifacts: results/viser_diagnostics_smoke/{render.png,observed.png,metrics.json}.
The same-frame initialized smoke verifies plumbing, not reconstruction accuracy
or live throughput. Live input RGB/depth handoff was connected; physical camera
interaction was not exercised by this smoke. User CUDA cache was not modified.

## Native Gaussian dock (left side)

Gaussian diagnostics now live in a native Viser panel docked on the **left**,
in the same browser page. No extra port or browser window is needed. Restart
the demo to load this layout. Enable **Render diagnostics** in the
**Gaussian diagnostics** tab for the whole render, observed RGB and depth error.
The **Part renders** tab shows all assigned parts as separate image cards, added
and removed as identities change. Images share the camera frame. The panel uses
Viser's native docking controls and can be moved by the user.

To opt into the experimental drawer split gate:

```sh
CONFIG=reprojection_split_residual_veto.yaml PORT=8090 OUT=results/author_demo_live \
  examples/multi_part/run_author_demo.sh live --depth 1
```

The gate was checked through recorded author_demo replay and can detect a
prismatic part earlier, but may produce an extra late part. It is not the
default and does not guarantee correct decomposition of a new recording.

Diagnostic target FPS controls render start spacing; achieved rate is limited by incoming frames and tracker/UI work. No render jobs are queued to catch up. User observed about 21 ms per diagnostic render on their live model; this is not a general benchmark.

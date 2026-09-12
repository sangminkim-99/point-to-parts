# Thin-slab turnover control

Owner: Claude (demo worker). Files: `scripts/sim/thin_turnover_control.py`,
`test/object/test_thin_turnover.py`, this document. Nothing else touched.

## Why

The user turns a **thin book over** in front of the fixed RealSense. Near
edge-on the mask collapses to a sliver a few millimetres thick and depth
support disappears. The honest behaviours a tracker must show there:

1. **hold, don't fabricate** — the pose is carried, not invented, while support
   is gone;
2. **say held, not observed** — exactly what the demo's uncertainty display
   (`held ≠ observed`, high-residual = "likely edge-on") is for.

No existing control passes through edge-on: `moving_laptop` yaws 20° in plane
and never loses its face. This control is the missing case, causal and
fixed-camera by construction (there is deliberately **no camera-orbit option**).

## Variants

| variant | GT | motion |
| --- | --- | --- |
| `rigid` | **1 part** — any split is false | thin slab (0.20×0.15×**0.006** m) flips 180° about an in-plane axis, passing exactly through edge-on; cosine-eased inside window 0.15–0.55, static before/after |
| `hinge` | **2 parts + 1 revolute** | body (0.02 m pages) + thin cover (0.004 m) hinged at the spine; the object flips 180° first, then the cover opens 120° (window 0.62–0.92; `--overlap` makes both simultaneous) |

Default azimuth is **180°**, chosen by measurement: after a 180° flip viewed
from azimuth 0 the opening cover fully occludes the body (body = 0 px through
the entire hinge window) — a control where one part is invisible tests nothing.
At 180° both parts stay visible throughout the hinge window.

## Conventions — identical to `render_partnet_sequence.py`

`rgb/` uint8 · `depth/` uint16 mm (0 = invalid) · `seg/` uint8 **part index,
255 = background** (oracle labels) · `meta.json` (intrinsics, parts, joints,
settings) · `poses.npz` (`T_cam_part (T,K,4,4)`, `joint_states (T,J)`,
`cam_poses`, `timestamps`, plus `mask_px (T,K)` for the edge-on profile).
Depth-noise dial (`--depth-noise`, `--depth-quant-mm`), seeded; the same
RGB/depth sub-0.1 px projection check as the renderer runs every frame.

**GT is eval-only; oracle mask labeling.** The tracker receives rgb, depth and
the binary **union** mask `seg != 255` only — per-part indices and `T_cam_part`
never enter tracking. `meta.json` carries a `gt_note` stating this, and
`examples/multi_part/prep_demo_input.py` / `eval_moving_root.py` consume the
layout unchanged (same seg convention).

## Commands

```sh
P=$HOME/miniconda3/envs/point2pose_model
ENV="env CUDA_HOME=$P PATH=$P/bin:$PATH \
     CPATH=$P/targets/x86_64-linux/include:$P/include \
     TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=2 \
     TORCH_EXTENSIONS_DIR=<worktree>/results/torch_extensions"

# render the two controls (reference copies already under results/thin_turnover_v1/)
$ENV python -m scripts.sim.thin_turnover_control --out results/thin_turnover_v1/rigid --variant rigid --frames 120
$ENV python -m scripts.sim.thin_turnover_control --out results/thin_turnover_v1/hinge --variant hinge --frames 120

# rigid moving-root evaluation with the existing helper (unchanged):
$ENV python -m examples.multi_part.eval_moving_root results/thin_turnover_v1/rigid

# or feed the demo/tracker via the oracle union mask:
$ENV python -m examples.multi_part.prep_demo_input results/thin_turnover_v1/rigid <track-dir>

# tests (8: 3 pure profile-math, 5 rendered-GT; ~2 s)
$ENV python -m pytest test/object/test_thin_turnover.py -q
```

## Reference sequences rendered (640×480, 120 frames, noise 0)

| sequence | union mask px | edge-on frame | GT verified |
| --- | --- | --- | --- |
| `results/thin_turnover_v1/rigid` | 40 368 → **3 094** (13×collapse) | 29 | world rotation 180.00° = script |
| `results/thin_turnover_v1/hinge` | 13 038 → **1 642** | 29 | per-frame relative body↔cover rotation ≡ scripted q to <0.1° |

The tests pin: output/seg/depth conventions, GT-equals-script for both
variants, a pronounced edge-on collapse inside the turnover window (min < 0.4×
max, `edge_on_frame` recorded in meta), both hinge parts visible through the
hinge window, and the `seg != 255` union convention.

## Baseline actually run: unmodified tracker on `rigid`

`eval_moving_root` on `results/thin_turnover_v1/rigid` (oracle union mask,
labeled as such in the output; no tracker changes; isolated build cache).
Full numbers: `results/thin_turnover_v1/rigid_baseline_analysis.json`;
tolerance rot 10° / trans 2 cm.

| metric | value |
| --- | --- |
| false splits | **0** (1 part throughout — correct) |
| observed coverage (all / flip window) | 0.86 / 0.66 |
| GT rot error, **observed** frames (n=103) | median **179.95°** |
| GT rot error, **held** frames (n=17) | median 45.2° (max 92.2°) |
| **false-fresh rate** (observed but out of tolerance) | **0.71** — first at frame 47 |
| recovery after the flip | **never** (latency ∞) |
| frame convention | verified: held frames 30–46 begin 1 frame after the mask minimum; meta `edge_on_frame` 29 = `mask_px` argmin 29 |

Reading, in the review's terms: the failure is **calibration, not
fragmentation or refusal**. The tracker correctly keeps one part and honestly
holds for 17 frames right after edge-on — but at frame 47 it re-locks onto the
**never-before-seen bottom face** near the old pose and reports *observed*
with ~180° GT error for the rest of the clip. Its held frames are actually its
*best* frames (median 45° vs 180° observed). So on this control the
interesting number is exactly the conditional one: pose error **given**
observed — not observed coverage itself, which at 0.86 tells you nothing about
whether those observations are right. Opposite-face re-association is the
mechanism (the ambiguity `doc/interactive_model_authoring.md` step 4
anticipates); a fix belongs to the tracking owner, not this harness.

## What else to measure with it (planning; not run here)

- **Rigid**: done — see the baseline above. The metric definitions it uses:
  `observed=False` at edge-on is **not by itself** a success metric, and 1.0
  observed coverage through edge-on is **not automatically** overclaiming — if
  the poses are accurate, claiming them observed is correct. What separates
  honest from fabricated tracking is conditional:
  - **GT pose error conditioned on observed vs held** — errors on frames the
    tracker calls observed should be small; held frames may drift, and that
    drift should be reported as held, not observed;
  - **calibration / false-fresh rate** — the fraction of frames claimed
    *observed* whose GT pose error exceeds tolerance (an observed claim with a
    wrong pose is the actual fabrication);
  - **recovery latency** — frames after the edge-on window until the pose is
    again both observed and within tolerance.
- **Hinge**: joint recovery (revolute, axis along the spine) with the motion
  split from the flip; `--overlap` for the harder simultaneous case. Late/no
  split during the flip is correct behaviour; a split *during* the rigid flip
  is a false split.

## Face appearance and the symmetry lock — measured

**Verified first:** the generator's slab used ONE constant box material — the
rendered object's masked colour statistics are identical to the digit before
vs after the 180° flip (mean BGR `[236.6 248.1 248.5]`, same std, frames
0/119). Top and bottom faces are genuinely indistinguishable, confirming the
keyframe report's suspicion at both source and pixel level.

`--faces distinct` (new, this generator + test only) breaks that: light top
half / dark-red bottom half of the *same rigid link* — geometry, seg and mask
unchanged; only appearance symmetry across the flip is removed. A rendered
test pins uniform ≈ identical (Δcolour < 8) vs distinct (Δ > 60).

**Scoring** uses root's `scripts/sim/symmetry_metrics.py` unmodified
(eval-only): predicted poses are made **absolute** before scoring —
`pred_abs(t) = rec(t) @ inv(rec(0)) @ gt(0)` — and the symmetry set is
**declared from the known cuboid geometry** (D2: 180° about x, y, z in the
link frame), never inferred from errors; strict RMS is always reported
alongside. Probe: `scripts/sim/face_matching_probe.py`; artifact:
`results/thin_turnover_v1/face_matching/face_matching.json`; revision
`f44f181`, identical seed/motion in both arms, oracle union masks.

| turnover, 120 fr | strict RMS med / final | symmetry-aware final (index) | obs cov | held frames |
| --- | --- | --- | --- | --- |
| uniform faces | 153.9 / **153.9 mm** | **1.0 mm** (2 = 180° about y) | 0.86 | 17 (f30–46) |
| distinct faces | 106.1 / **153.9 mm** | **0.6 mm** (2 = 180° about y) | **1.00** | 0 |

Reading:

1. **The failure is exactly the symmetry partner.** In both arms the final
   pose is the GT composed with the *declared turnover-axis flip* to within
   1 mm surface RMS — 153.9 mm strict error, ≤1 mm symmetry-aware. This is a
   quantitative identification of opposite-face re-association, not drift
   (uniform: 68/120 frames are strict-bad but symmetry-explained; the index
   locks to the y-flip at f46 and never leaves).
2. **Distinguishable faces do NOT fix it.** The distinct arm ends in the same
   flip-lock (0.6 mm to the symmetry partner). Consistent with this
   pipeline's design: registration scores geometry only (colour in the energy
   was measured and rejected earlier in the project), and TAPIR's tracks die
   at edge-on regardless of what colour the new face is — so appearance
   distinctness alone cannot veto the geometric re-lock. Any fix needs an
   appearance *check* at re-association time (e.g. keyframe colour
   consistency against the model), which is tracker-owner territory.
3. **The distinct arm was also worse mid-flip, in a confounded way**: strict
   error is already ~107 mm at f30 (uniform: 6 mm) and it never holds
   (coverage 1.00, zero held frames — worse calibration than uniform's honest
   17-frame hold). The dark, low-contrast underside plausibly degrades
   keypoints as it rotates into view — but this confounds *distinctness* with
   *darkness/texture quality*, so it is reported as an observation, not a
   causal claim.

## Stored-appearance evidence vs the mirror pose (oracle diagnostic)

Uses root's `examples/multi_part/appearance_evidence.py` unmodified (revision
`c973e47`): normalized RGB L1 over depth-supported (±12 mm), z-buffered stored
Gaussian centres — a **diagnostic**, not a rasterized loss, and nothing here
feeds tracking. Probe: `scripts/sim/appearance_probe.py`. Protocol: freeze the
dense model's geometry+colours at the last pre-flip frame (f16; the model has
seen top + sides, never the bottom), then score the tracker's **actual**
mirror-locked poses and a **labeled oracle** hypothesis
`gt(t)·gt(0)⁻¹·rec(0)` (GT enters only through this oracle; never a runtime
hypothesis). Both arms share seed/motion; sanity anchor at the freeze frame:
both hypotheses agree (n≈4970, L1≈2e-9, support ratio 1.000).

| post-flip steady state | tracked = mirror pose | gt oracle |
| --- | --- | --- |
| uniform: support / mean L1 | **4937 / 0.000014** | 363 / 0.353 |
| distinct: support / mean L1 | **4957 / 0.289** | 363 / 0.347 |

**Support asymmetry is geometric, and disqualifies the ranking.** Under the
true pose the frozen (now-hidden) surface lies `thickness / sin(elevation)`
≈ 14–19 mm behind the observed face *along the ray* (measured dz percentiles
13.8–18.8 mm) — outside the ±12 mm gate — so true-pose support collapses to a
363-point fringe (7%) whose colours are silhouette-contaminated. The mirror
pose, by construction, coincides with the observation (dz within ±1 mm, ~full
support). Per the metric's own contract this unequal-support comparison is
**not a fair ranking**, and it is not treated as one: the oracle's 0.35 says
"fringe samples of a never-seen face", not "worse pose".

**Answer to the question — can stored appearance reject the mirror pose under
distinct materials, despite illumination?**

- **As a ranking against the true pose: no**, in both arms — the depth gate
  structurally favours the mirror pose on a thin object at 25° elevation, and
  the frozen model has never seen the revealed face, so even the true pose
  scores badly on what little support it keeps.
- **As an absolute self-consistency check: yes, cleanly, on this sim.** The
  mirror lock carries a persistent **L1 ≈ 0.289 on ~full support (4952–4973
  samples)** in the distinct arm, vs **0.000014 in the uniform arm under
  identical lighting and geometry** — so the signal is albedo, not
  illumination (four orders of magnitude apart; a 0.1 threshold separates
  with huge margin). The uniform arm shows the converse: a mirror pose that
  appearance cannot flag at all.
- **Mid-flip colour error is NOT usable**: during the flip even the oracle
  pose reads L1 0.38–0.57 (never-seen surface + near-edge-on shading), so a
  high score there means "unmodelled view", not "wrong pose". Any consumer
  must wait for support to recover (the tracked support curve itself shows
  the capture: it collapses to 44 points at f30 and snaps back to 4400 at the
  f47 re-lock) and compare against an expected-noise baseline, on
  **full-support frames only**.

Caveats recorded in the artifact: oracle labeling, unequal support, the
never-seen-face limitation, and that this sim's illumination is simple —
real-glare robustness is untested. Artifacts:
`results/thin_turnover_v1/appearance_{rigid,rigid_distinct}/appearance_probe.json`
(+ `_arrays.npz`, final rgb/depth/mask for the dz breakdown).

### Paired evidence (`05f594a`) and the signal-vs-recovery split

Rerun with root's `paired_appearance_evidence` (same visible stored IDs under
BOTH poses, min 20; ~340–360 paired samples on all 54 post-flip frames). New
artifacts `appearance2_{rigid,rigid_distinct}/` — the earlier outputs are
preserved untouched.

| post-flip, paired (same IDs) | tracked L1 | oracle L1 | gain frames favouring oracle |
| --- | --- | --- | --- |
| uniform | 0.000019 | 0.353 | **0 / 54** |
| distinct | 0.183 | 0.346 | **0 / 54** |

- **Even the fair, same-sample comparison cannot recover the true pose**:
  the paired subset can only contain pre-flip-seen surface, which the mirror
  pose projects coherently and the true pose projects onto
  silhouette-contaminated pixels. 0/54 frames favour the oracle in either
  arm. Appearance evidence over a pre-flip snapshot **cannot recover the
  unseen backside** — paired or not.
- **But it signals inconsistency cleanly, under distinct materials.** The
  tracked absolute L1 distributions (good pre-flip vs mirror-locked
  post-flip): uniform — good max 6.3e-4 vs bad max 2.5e-5, **no signal at
  all**; distinct — good max **1.3e-6** vs bad p5 **0.289**, five orders of
  magnitude, zero overlap: any threshold in [1e-5, 0.28] flags 54/54 bad
  frames with 0/17 false alarms on this sim. Signal, not recovery: the
  diagnostic can say "this pose contradicts stored appearance", never which
  pose is right.

## Limitations

- Untextured coloured slabs (like the laptop control): SuperPoint keypoints come
  from silhouette/edges, so keypoint starvation is harsher than on a printed
  cover — harder in that dimension. Real books can be harder in others (glossy
  covers defeating stereo depth, deformable pages, sensor noise), so **no
  overall easier/harder ordering is asserted**; this control isolates the
  geometric edge-on effect with clean depth.
- Part frames are link frames (body centre / spine), not centroids — same
  convention as the renderer; only relative/first-frame-aligned motion is
  meaningful for evaluation.
- No physics stepping: poses are scripted kinematics (exact GT, no gravity sag).
- The cover is visible only as a sliver right as it starts opening (it *is*
  edge-on then) — inherent to the geometry, not a rendering artefact.

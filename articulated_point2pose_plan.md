# Articulated Point2Pose — Project Plan

Working title: **Part2Pose: Online Model-Free Articulated Object Tracking, Reconstruction, and Part Discovery from 2D Point Tracks**

## 1. Goal and thesis

Extend Point2Pose (Lin et al., 2026) from rigid objects to articulated objects, online and causally, from a single RGB-D stream, with no CAD model, no category prior, and no assumption about the number or type of joints.

Central claim: the rigid-body assumption in Point2Pose enters only at the frame-to-map registration step. Its sequential RANSAC already extracts multiple rigid-motion consensus sets. Treating persistent secondary consensus sets as *parts* — and coupling them through an estimated kinematic graph — yields articulated tracking, part segmentation, and joint estimation from essentially the same machinery.

Deliverables, in priority order:
1. Per-part 6D pose tracking with occlusion recovery (the Point2Pose strength, preserved).
2. Online joint discovery and estimation (type, axis, state) — the headline contribution, and what makes (1) well-conditioned for small textureless parts.
3. Dense part segmentation and per-part reconstruction as a rendering-based byproduct.

Explicit non-goal: semantic parts. Parts are defined by motion. A drawer that has never been opened is one rigid body with the cabinet, and that is the correct answer.

## 2. System design

### 2.1 State representation

Per tracked object *i*:
- A set of parts `P_i = {P_0 (base), P_1, ..., P_K}`, each with
  - a keypoint map in its own part frame (as in Point2Pose §3.2),
  - a dense model in its part frame (TSDF in v0, Gaussians in v1 — see §3),
  - keyframe pose history.
- A kinematic graph: edges `(parent, child, joint)` with joint parameters
  - twist `ξ ∈ se(3)` (Plücker axis + pitch) in the parent frame,
  - joint type ∈ {revolute, prismatic, screw, unknown},
  - joint state `θ_t` per keyframe, with limits accumulated from observed range.
- Per-keypoint and per-Gaussian **part-assignment log-odds** accumulated over time (not hard labels).

### 2.2 Per-frame pipeline

1. **Point tracking** — unchanged: one tracker pass over the aggregated query set of all objects and parts.
2. **Hypothesis generation** — sequential RANSAC + SVD over all tracked keypoints of object *i*, producing rigid-motion hypotheses `H = {T_1, ..., T_H}`. Keep this as-is from Point2Pose.
3. **Part-constrained hypothesis generation** — for each existing part with a known joint, additionally generate the *1-DoF* hypothesis: solve for `θ_t` only, along the joint axis. This is far better conditioned than free 6-DoF from 3–5 tracks and is the main reason to estimate joints.
4. **Dense assignment and selection** (§3) — score each hypothesis against the masked depth (and color) observation; assign geometry to hypotheses; select one pose per part.
5. **Part lifecycle** (§2.3) — spawn / merge / retire parts.
6. **Refinement** — dense refinement of each part pose (TSDF-ICP in v0, differentiable rendering in v1).
7. **Graph optimization** — on new keyframe, joint optimization of part poses, keypoints, and joint parameters (§2.4).
8. **Model update** — fuse the frame into each part's dense model in the part frame; sample new keypoints *per part* using the dense part mask.

### 2.3 Part lifecycle (lazy, evidence-driven)

- **Spawn.** A candidate part is created when, over a verification window (reuse the pending-point mechanism), (a) the single-best rigid hypothesis leaves a spatially coherent residual blob in the dense scoring, and (b) a secondary RANSAC consensus set persists with consistent relative motion to the base. Both signals are required; the residual blob alone fires on depth noise, the consensus set alone fires on aliasing.
- **Assign.** New keypoints and Gaussians inherit the part whose hypothesis best explains them, entered as soft log-odds.
- **Merge.** Two parts whose relative pose has been static (within noise) for longer than a threshold are merged (this handles spurious splits).
- **Retire.** Parts unobserved for long stay in the graph (occlusion recovery must work per part); nothing is deleted.
- **Joint fitting.** Once a part has ≥N keyframes of relative motion with sufficient angular/linear spread, fit a screw from the sequence of relative transforms (closed form from relative SE(3) motions, then refine in the factor graph). Model selection revolute/prismatic/screw by residual with a complexity penalty; "unknown" until confident.

### 2.4 Factor graph

Extend the per-object graph of Point2Pose §3.4 with:
- Part pose variables `X_{k,m}` for part *k* at keyframe *m* (inverse-pose parameterization retained).
- **Screw factor** per joint and keyframe:
  `T_{child,m} = T_{parent,m} · Exp(θ_m ξ) · T_{child,0}`,
  with `ξ` a single global variable per joint and `θ_m` per keyframe. Huber-robust.
- Optional **joint-limit** and **smoothness** factors on `θ_m`.
- Observation factors are unchanged but indexed by part; keypoint-to-part assignment is fixed at the log-odds argmax at optimization time (soft assignment inside the optimizer is a later refinement).

Before a joint is confidently fitted, the child part is simply a free rigid body attached only by observation factors. Fitting the joint later back-constrains earlier keyframes.

## 3. Dense model and rendering-based part assignment

### 3.1 The proposed idea, stated precisely

- Represent each object as a set of 3D Gaussians, initialized by back-projecting masked depth (position from depth, scale from local pixel footprint, color from RGB, opacity ~1).
- Each Gaussian carries part-assignment log-odds over the current parts.
- For a new frame, for each hypothesis `T_h` (from §2.2 steps 2–3):
  - transform the Gaussians *whose assignment is uncertain, or which belong to the part T_h is a candidate for* by `T_h`,
  - rasterize depth (and optionally color) with the z-buffer,
  - compute per-pixel residual against observed depth (and color).
- Per-pixel min-residual across hypotheses gives a dense hypothesis map. Splat it back onto Gaussians (each Gaussian's pixels vote) and *add* to its log-odds.
- Pixels with high residual under **all** hypotheses are unexplained geometry → new Gaussians (this is how drawer interiors and previously self-occluded surfaces enter the model).

This is an EM loop: E-step = rendering-based assignment; M-step = pose refinement per part by differentiable rendering on its assigned Gaussians, plus periodic Gaussian parameter refinement.

### 3.2 Why it is better than per-part TSDF scoring

- Visibility: a hypothesis that moves a part in front of the base must occlude it; rendering enforces this, point-wise TSDF lookup does not.
- Photometric evidence: on geometrically flat but textured parts (drawer fronts, printed panels) depth cannot distinguish "moved" from "static", color can.
- Differentiable refinement of part pose *and* geometry in one framework; the same model serves reconstruction output.
- Residual maps double as a part-discovery signal (§2.3 spawn condition (a)).

### 3.3 Known hazards and mitigations

| Hazard | Mitigation |
|---|---|
| K renders per frame per object | K is small (2–4). Depth-only rasterization, no SH, at half resolution for scoring; full-res only for the selected hypothesis' refinement. Render only uncertain/candidate Gaussians. |
| Aliasing: flat/symmetric regions explained equally by several hypotheses | Never decide per frame. Log-odds accumulation with a decay; spatial smoothness prior on assignment (Gaussian kNN graph); kinematic prior (1-DoF hypotheses). |
| Depth noise makes Gaussians from a single frame poor | Same pending mechanism as keypoints: Gaussians are provisional until confirmed by re-observation; optimize scale/opacity; prune low-opacity and floaters. |
| Drift between Gaussian model and keypoint map | Both live in the same part frame and share the same keyframe poses from the factor graph; re-anchor Gaussians after each graph optimization. |
| Sim-to-real depth artifacts (RealSense edge bleeding at part boundaries) | Erode masks at depth discontinuities before creating Gaussians; use confidence-weighted depth residual. |

### 3.4 Staging

- **v0 (baseline, ~weeks):** per-part TSDF. Minimal change to existing code; gives a working articulated tracker fast and a baseline to compare the Gaussian version against.
- **v1:** Gaussian model with rendering-based assignment and differentiable refinement.
- **Reconstruction output:** mesh per part via TSDF fusion of confirmed Gaussians (or marching cubes on a density field), so evaluation uses the same Chamfer protocol as the paper.

## 4. Evaluation

### 4.1 Datasets

- **Simulation:** PartNet-Mobility assets in Isaac Lab (the authors' existing rendering pipeline). Scripted articulation trajectories with hand/arm occluders; exact ground truth for part poses, joint axes, states, and part masks. Include (a) single-joint, (b) multi-joint (cabinet with two drawers, laptop + lid), (c) two articulated objects simultaneously, (d) complete occlusion and re-entry mid-articulation.
- **Real:** OptiTrack markers on *each part* (the YCBMultiTrack rig). Objects: drawer box, cabinet door, laptop, scissors, pliers, storage box with lid, a textureless drawer as a deliberate hard case. Sequences: human hand and robot arm manipulation, full occlusion and re-entry with joint state changed while occluded.

### 4.2 Metrics

- Per-part ADD / ADD-S AUC (same protocol as the paper).
- Joint axis: angular error (deg), positional error for revolute (cm).
- Joint state error (deg / cm) over time.
- Part discovery: time-to-detect after first motion; false split / false merge counts.
- Dense part segmentation: per-frame IoU against GT part masks.
- Per-part reconstruction: Chamfer distance.
- Runtime and memory vs. number of parts.

### 4.3 Baselines

- Point2Pose treating the object as rigid (shows what breaks).
- Point2Pose run independently per part with GT part masks (upper bound for tracking; shows the value of the kinematic prior when masks are given).
- Offline articulated reconstruction methods (PARIS / Ditto / ArtGS family) on the final two-state pair — different problem setting, but reviewers will ask. Verify the current state of this literature before writing related work; it has moved quickly.
- FoundationPose per part with GT part meshes as the CAD-based reference, as the paper does for rigid objects.

### 4.4 Ablations

- 6-DoF vs. joint-constrained (1-DoF) hypotheses on textureless parts.
- TSDF scoring vs. rendering-based assignment (depth only vs. depth + color).
- Per-frame hard assignment vs. accumulated log-odds.
- Spawn criteria: consensus-only vs. residual-only vs. both.

## 5. Phases and milestones

**Phase 0 — Setup (2 weeks)**
- Get Point2Pose running on YCBMultiTrack; profile the RANSAC and TSDF components.
- Build one PartNet-Mobility drawer sequence in Isaac Lab with GT.
- Milestone: reproduce a paper number; one articulated sim sequence with GT.

**Phase 1 — Multi-part tracking, v0 (4–5 weeks)**
- Refactor object → parts; per-part keypoint maps and TSDFs.
- Part spawn/merge from persistent secondary RANSAC consensus sets.
- Milestone: track a two-part object in sim with per-part ADD; part discovery timing reported.

**Phase 2 — Joint estimation (3–4 weeks)**
- Closed-form screw fit from relative motions; screw factor in the graph; 1-DoF hypotheses.
- Milestone: axis error < 5° and state error < 5° / 1 cm on sim; demonstrate the textureless-drawer case working only with the joint prior.

**Phase 3 — Gaussian model and rendering-based assignment, v1 (5–6 weeks)**
- Gaussian init from depth; per-hypothesis rasterization; log-odds accumulation; unexplained-pixel spawning; differentiable refinement.
- Milestone: dense part IoU and Chamfer reported; ablation vs. v0.

**Phase 4 — Real data (4 weeks, overlapping with Phase 3)**
- Multi-part mocap rig; collect ~15 sequences; time sync and calibration as in the paper.
- Milestone: real-world tables for tracking, joint, segmentation.

**Phase 5 — Writing and release (3–4 weeks)**
- Paper, code, dataset (name it as the articulated extension of YCBMultiTrack).

Total ≈ 5 months. The MIT visit is scheduled to end in September, so Phase 4 (which needs the OptiTrack rig) should either be pulled forward or planned as a lab-side collaboration with remote integration; decide this with the original authors early.

## 6. Risks

- **Textureless parts** remain the dominant failure mode; the joint prior mitigates but cannot rescue a part that never yields ≥3 tracks. Report honestly; propose depth-feature tracks as future work.
- **Spurious splits** on non-rigid occluders (hands) leaking into masks. Mask erosion plus the requirement that a part persist across keyframes.
- **Scope creep**: three deliverables. If time runs short, drop v1 (Gaussians) and ship v0 with joint estimation; that is already a complete paper.
- **Novelty overlap** with articulated Gaussian-splatting work. The defensible niche is *online, causal, model-free, multi-object, occlusion-recovering* — keep every experiment stressing those four properties.

## 7. Immediate next steps

1. Talk to Tzu-Yuan Lin and Ho Jae Lee: codebase state, known brittleness, whether an articulated extension is already planned, authorship expectations.
2. Prototype the cheapest experiment that tests the thesis: run the existing sequential RANSAC on a sim drawer sequence and check whether the second consensus set is stable across frames. If yes, the whole plan is de-risked; if not, the spawn logic needs more work before anything else.
3. Set up the PartNet-Mobility → Isaac Lab pipeline.

# Joint tracking versus observed-surface evidence

2026-09-10. Matched controls isolate the existing scalar joint-pose override.
All variants use `guarded_incremental` and causal RGB-D/union masks. GT is used
only in saved-trace evaluation. No default has changed.

The current joint search minimizes sparse 3D correspondence residual over a
scalar grid. A candidate may replace the free pose when its residual is within
1.3 times the free residual. This does not independently establish agreement
with observed surfaces. The new experimental `joint_reprojection_gate` checks
both poses on the same sample of at most 1024 stored Gaussian centers: joint pose
must have at least 25% depth support, and support-minus-contradiction cannot fall
more than 0.02 below the free pose. At least 40 stored points are required.
Depth agreement uses the existing 12 mm tolerance; hidden surfaces are neutral.
The existing sparse residual test still applies. This is an engineering ablation
of standard geometric validation, not a novel method or joint-search replacement.

`ablate_reprojection.py` exposes `guarded_no_joint_track` and
`guarded_joint_reprojection`. These compare respectively no scalar override and
a surface-validated override against the unchanged `guarded_incremental` control.
Artifacts, commands, working patches and traces live in:

- `results/confidence_controls_v1` (joint override reference)
- `results/joint_tracking_ablation_v1` (no joint override)
- `results/joint_reprojection_v1` (experimental surface gate)

## Clean hinge

The reference has 47 qualifying lid observations, one ID and 434.2 mm / 43.0
 degrees own-observed median error. Disabling joint tracking gives 41 observations,
two IDs (one switch), and 118.7 mm / 11.36 degrees. On the common 41-frame subset,
reference is 432.4 mm / 43.0 degrees versus 118.7 mm / 11.36 degrees with no joint
tracking. Alignment is fixed separately for each ID; the replacement ID receives
its own gauge. Thus the lower error includes a reset and reduced coverage.

The surface gate accepts zero joint overrides on this clip and reproduces the
no-joint result. It does not establish a benefit over simply disabling the
constraint. Both split again at frame 110; the final lid ID lasts only 11 frames.
Do not claim 90 mm final-ID error as a whole-sequence result. Joint constraints
currently prolong a bad pose trajectory; rejecting them alone does not solve
opposite-face identity or geometry recovery.

## Drawer and outcome

All three variants have 76 qualifying moving-part observations, no ID switch,
and split at frame 45. Common-frame median errors (mm / degrees) are reference
5.11 / 0.67, no joint override 4.09 / 0.50, and surface gate 5.11 / 0.67. The gate
retains 31 joint-tracked part-frames versus reference 32. The two controls therefore
show selective behavior (hinge rejected, drawer mostly retained), but do not show
that retaining the constraint improves pose accuracy. All three retain 97.6%
final sparse purity and two GT groups.

Timing is recorded per run in tracking.log. No-joint step medians were 96.3 ms
(hinge) and 78.3 ms (drawer), still short of a 30 Hz pipeline. These short runs are
not a controlled runtime study. The experimental gate is opt-in. Its thresholds
need broader testing, and hidden or incomplete stored surfaces can reject a
correct pose. Next focus on discovering/recovering the missing reverse-face
surface and improving pose initialization rather than promoting this veto.

Relevant tests (60 passing) include unsupported/occluded candidate rejection,
preference for matching depth, insufficient geometry, existing pose hypotheses,
confidence gates, simulation geometry and URDF kinematics. The 3-way comparison
JSON files are `results/joint_reprojection_v1/*_comparison.json`.

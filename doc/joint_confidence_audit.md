# Joint-axis confidence audit

2026-09-10. The existing model compares rigid, prismatic and revolute candidates
using noise-normalized pose residuals and a BIC-style score. Its broad approach
follows [Sturm, Stachniss and Burgard](https://arxiv.org/abs/1405.7705), which learns
kinematic relationships from noisy part poses. The implementation also uses
engineering choices (geometry prior, excitation thresholds, bootstrap axis
stability). This audit does not claim a reproduction of the paper or a new method.

## Corrected confidence calculation

The bootstrap used `std(acos(axis_sample dot mean_axis))`. For samples at +30
and -30 degrees, every distance from the mean is 30 degrees: the standard
deviation of those distances is zero despite large axis dispersion. It could
therefore incorrectly report a confident axis. Missing bootstrap fits also
mapped NaN dispersion to `axis_ok=1`, another unwarranted certainty.

Now report the RMS angle between each normalized bootstrap axis and the fitted
axis, with absolute dot products so axis sign is irrelevant. Insufficient samples
(fewer than four) give zero axis confidence and invalidate an articulated fit's
confidence gate. Rigid/disconnected models do not bootstrap a fictitious slider
axis. The compatibility key `axis_std_deg` now contains this RMS value; the
explicit key `axis_rms_deg` and `axis_bootstrap_samples` document the semantics.
This is a stability diagnostic, not a calibrated posterior or confidence interval.
It does not fix biased poses, joint type mistakes, or pivot-location uncertainty.

## Validation

Three regression tests check symmetric 30-degree dispersion, missing bootstrap
fits, sign equivalence and an actual clean prismatic fit. The symmetric case now
reports 30 degrees and axis confidence below 0.03; missing evidence reports zero
confidence. Existing kinematics/export tests remain passing.

A CPU diagnostic replayed 27 growing prefixes (8 through 112 observations) of
`results/urdf_validation/drawer_joints.npz`, containing saved estimated relative
poses. Types were unchanged across all prefixes. At 112 observations both select
prismatic; dispersion changes from 0.10094 to 0.17198 degrees and confidence from
0.98746 to 0.97873, retaining valid status. Full results:
`results/joint_confidence_v1/drawer_prefixes.json`.

This replay uses default JointModel settings; the saved file does not preserve
online geometry/noise floors. It is not an end-to-end tracking ablation. BIC
scores, fitting and inferred axis geometry are untouched. Runtime gates consuming
confidence may now act more conservatively, so the next simulation runs should
check discovery delay, recovery coverage and exported joints before attributing
tracking improvements to this fix. No real-time performance claim is made.

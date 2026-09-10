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

## End-to-end controls after the confidence fix

At commit `4a7eca3`, reran `guarded_incremental` on clean hinge and drawer using
`ablate_reprojection.py --trace`. Each run retains its command, source commit,
working patch, video, log and pose trace in `results/confidence_controls_v1`.
Compared against `results/pose_guarded_v1` using `compare_pose_traces.py`.
All saved pose matrix entries and observed flags are identical in both cases.

| Case | Moving-part observed / 119 | Shared median mm / degrees | Step median / p90 ms |
|---|---:|---:|---:|
| Drawer | 76 | 5.11 / 0.67 | 81.0 / 104.2 |
| Clean hinge | 47 | 434.21 / 43.00 | 96.9 / 137.1 |

The two-part splits remain at replay frames 45 and 74 respectively. Final joint
types remain prismatic and revolute. The reported tracker step throughput is
12.3 and 10.3 FPS; this excludes camera acquisition and full application costs
and does not meet a 30 Hz target. These runs validate no regression on these two
clips, not improvement or robustness on other objects.

The hinge still had a historical `CONTROLLABLE` event despite its bad trajectory.
That event only checked type-times-axis confidence, ignoring validity and
excitation. The diagnostic event now also requires an articulated kind, valid
confidence, sufficient excitation and finite confidence. Logs say
`JOINT ESTIMATE READY` / `first joint-estimate readiness`, and label axis spread
as RMS rather than a +/- interval. The legacy `controllable` attribute is retained
for compatibility. This is a historical event, not current confidence or verified
robot controllability; it cannot authorize physical manipulation. Five regression
cases cover these event gates. This logging/gating change was made after the
above runs; it does not change tracker pose updates or joint fitting.

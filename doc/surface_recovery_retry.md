# Retrying surface recovery for contradicted feature poses

2026-09-10. The recovery path previously skipped any part with a fresh sparse
pose, even when that pose disagreed with depth. An optional
`recovery_contradicted_pose` now also searches for such parts when the current
stored-surface reprojection has at least 25% contradiction and less than 25%
support. It retains the existing trusted-parent/joint requirements, distinct
surface-match margin and two-frame temporal confirmation. Hidden geometry alone
is not contradiction. Default behavior remains unchanged.

Accepted recovery retains the part ID and marks the pose as depth-recovered,
not an independent sparse observation for online joint fitting. The recovery
search uses existing joint geometry and observed depth; it does not infer a new
axis from its own prediction. `recovery_diagnostics` counts branch outcomes, and
replay traces now store a separate `recovered` flag alongside `observed` and
`present`. Old trace writers without recovery flags remain supported.

Two tests verify that a contradicted observed pose can be recovered without
changing joint observations, and that a supported observed pose is not searched.
The full relevant suite passes 62 tests. The extra search is an engineering
ablation of existing geometric recovery, not a new method.

## Initial observations

`results/recovery_retry_v1` contains clean hinge and hinge+orbit runs with the
option enabled. Clean hinge is exactly unchanged in saved poses/observed flags:
42 known-part frames are not geometrically contradicted enough, three fail joint
or parent trust, and only one search passes to temporal confirmation. It never
confirms. No recovery success should be claimed from this clip.

Combined motion has four depth recovery part-frames versus three in the earlier
guarded run; only trace frame 110 has a changed pose. Qualifying sparse coverage
and final reported medians are unchanged. This prompted a fresh matched run
rather than attributing differences across code revisions to the new option.

## Matched result: an extra recovery is a false recovery

Fresh runs in `results/recovery_retry_v2/laptop_hinge_orbit` use identical code
and differ only by the retry option. Baseline recovers lid frames 111, 112, 115;
retry adds frame 110. Both retain 31 independent sparse observations. With one
fixed alignment at the lid ID's first pose, frame 110 changes from a held pose
at 75.09 mm / 5.39 degrees error to a recovered pose at **444.64 mm / 44.82
degrees**. The added recovery is wrong. Overall medians conceal this single-frame
regression. The numeric audit is `results/recovery_retry_v2/recovery_comparison.json`.

Tracker step medians are 105.7 ms baseline and 104.2 ms retry (single short runs,
not a speed claim). The new option stays disabled by default. This result argues
against merely widening recovery eligibility or counting recoveries as success.
A known joint plus whole-object depth support is insufficient to establish part
identity. Next inspect whether candidates match a different part's surface and
whether stored canonical geometry/axis bias explains ambiguity. That is a
hypothesis, not a confirmed cause; no GT labels should enter recovery scoring.

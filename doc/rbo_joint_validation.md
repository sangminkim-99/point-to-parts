# RBO joint validation: matched geometry, incomplete graph

Reviewed worker report:
`results/frozen_gate_rbo_v1/rbo_joint_geometry_eval.md`.
Cardboardbox01 baseline and frozen-fallback both report a matched revolute lid
axis error of 3.9 degrees and line distance 0.7 cm. The corresponding lid
trajectory error is 75.6 mm / 7.56 degrees under the evaluator's alignment.
These results do not certify robot-ready geometry.

The reference axis is fitted to mocap relative poses using the same JointModel
estimator; only joint type comes directly from the RBO object specification.
Axes are compared after transport to camera coordinates. SAM2 masks use a
GT-derived initialization box, with predicted propagation thereafter.

Ikeasmall02 reports two predicted prismatic rows against the SAME rb2 joint,
with 0.3–2.3 degree errors, while another drawer has no matched joint row.
The overall graph is therefore incomplete/oversegmented despite small matched
axis errors. The evaluator now reports unique matched GT joints and duplicate
rows separately, and individual rows retain predicted part/parent attribution.
This does not alter the existing row-weighted angular summary; consumers must
not interpret row count as joint recall.

Additional evaluation caveats: the current evaluator checks a GT child has a
spec joint, but does not explicitly require the mapped parent to equal the
spec parent. Pose logs filter by final part count, not persistent identity.
These need auditing before a graph-level accuracy claim. Existing numbers are
retained under their original protocol, not retrospectively reinterpreted.

The evaluator now also attaches the spec parent and a tri-state
`parent_matches_spec` flag to each row. Summaries distinguish correct, wrong,
and unknown-parent rows. Existing angular metrics remain unfiltered for
backward comparison; correct topology must be checked separately. Historical
rows without the flag are unknown, never implicitly correct. This implements
parent attribution but does not resolve the pose-log persistent-ID limitation.

Replay now supplies per-frame identity logs to joint evaluation. Frames are
selected by the evaluated child/parent IDs, and the parent's pose is retrieved
by its ID rather than current list index. Filtered evaluation windows retain
frame-keyed IDs. Rows state whether identity alignment was supplied; legacy
external callers without it retain the old count-based behavior. Published
historical numbers are not silently replaced and require a new evaluation to
claim this stronger protocol.

## Identity-aligned cardboard rerun at 6c980e7

Worker report `results/frozen_gate_rbo_v1/rbo_joint_identity_rerun.md` confirms
cardboard fallback: one matched revolute row, 3.900-degree axis error, 0.71-cm
line distance, identity alignment enabled, rb0 parent matching the spec, no
duplicate GT rows. Same reported precision as the old protocol. This applies
to the rerun fallback arm only; ikea and other historical arms retain their
original protocol labels. The standard evaluator formatter now prints parent
agreement, identity alignment, and duplicate/unique counts directly.

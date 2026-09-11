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

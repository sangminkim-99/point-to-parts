# Thin-part turnover and rendering diagnostics

User priority: inspect Gaussian model/rendering in Viser and handle thin books
turning over. Preserve the drawer fix as the immediate tracking task.

## Established baseline and scope

BundleSDF (CVPR 2023) combines a posed memory-frame pool, pose-graph optimization
and an online object field for rigid object reconstruction/tracking:
https://openaccess.thecvf.com/content/CVPR2023/papers/Wen_BundleSDF_Neural_6-DoF_Tracking_and_3D_Reconstruction_of_Unknown_Objects_CVPR_2023_paper.pdf
Use keyframe memory/relocalization as an established baseline; importing its
neural field or claiming a new relocalization method is not required.

## Proposed causal handling (not implemented)

Separate the geometric surfaces from persistent rigid-part identity. Front and
back observations may be separate surface patches but share a part pose. Nearby
opposite faces must not be merged solely on Euclidean proximity, especially
when thickness approaches depth noise; use viewing direction/normal and history.

Near edge-on, depth support and feature overlap can vanish. Mark pose as uncertain,
hold or predict for visualization only, and suspend destructive fusion/carving,
new split decisions and joint fitting from unsupported poses. Preserve previous
surfaces. A motion prior or learned joint constraint may propose hypotheses but
is not independent evidence confirming them.

After turnover, compare current RGB-D against prior keyframes and accumulated
part appearance/geometry; jointly check support, robust reprojection and motion
continuity. Evaluate multiple plausible poses when symmetric/textureless views
are indistinguishable. Do not manufacture certainty or a new part solely because
new appearance arrived. A fully unseen back face without overlap can remain
unresolved; guide the user to show an oblique view or revisit a known face.

Distinguish rigid closed-book turnover (must remain one part), opening a rigid
cover relative to pages (joint), and bending individual pages (outside rigid-part
assumptions). No claim of general deformable-page reconstruction.

## Matched experiments / possible contribution

Fixed camera: (1) rigid thin slab 180-degree turnover, (2) hinged thin cover,
(3) joint held fixed with whole-object rotation, (4) combined articulation and
rotation. Sweep thickness, texture, edge-on duration and speed. Include repeated
front-back-front visits. GT poses/labels stay evaluation-only; oracle object
masks explicitly labeled. Compare last-frame tracking, keyframe-memory baseline,
plus uncertainty-aware map preservation, plus relative-joint constraints, plus
optional Gaussian render pose refinement. Report ID switches/false splits,
observed pose coverage vs held/predicted coverage, recovery latency, pose/joint
error, historical front/back surface retention, and runtime.

Potential contribution is online articulated identity and model continuity
across loss of visible surface overlap, if improvements survive these ablations.
It is not established novelty merely because a Gaussian renderer is used.

## Viser assignment

Demo worker: optional actual Gaussian RGB rendering next to observed RGB, depth
residual/support diagnostics; renderer/refiner enabled state, frame freshness,
counts and timing. Point cloud and Gaussian image must be named distinctly.
Run rendering in the tracking/worker path, not GUI callbacks. Avoid adding
per-frame cost when diagnostics are off. Held poses must be visibly identified.

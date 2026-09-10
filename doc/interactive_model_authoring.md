# Interactive online articulated model authoring

Design proposal, 2026-09-10. User requires a fixed camera with a moving object:
whole-object translation/rotation is essential to expose its surfaces. Interpret
"authorizable" as user-reviewable/editable model creation for now. This proposal
does not claim an implemented interface or reliable moving-root tracking.

## Experience and first demonstration

The user teaches an object by moving it while its editable articulated model
grows beside the RGB-D view. The system suggests the next useful demonstration,
keeps provisional hypotheses visible, and lets the user approve or correct them.

1. Select the object and optionally its reference body. Name it if useful.
2. Turn the whole object slowly, initially keeping joints at roughly one
   configuration. Grow observed part-local clouds and acquire overlapping views.
3. Move one joint. Suggest holding the reference body steady when ambiguity is
   high, but do not require a permanently stationary object. Show a proposed
   part separation and hinge/slider overlay, with accept/reject/undo controls.
4. Turn the object again at the new articulation state. Attach newly visible
   surfaces to persistent part identities using tracked overlap, geometry and
   joint constraints. If association is ambiguous, keep a provisional fragment
   and request an intermediate/previous view or user correspondence.
5. Scrub a joint slider on a copy of the model to preview unobserved
   configurations. Keep the live tracked state separate from this preview.
6. Test the prediction against the next real motion using RGB-D reprojection.
   Save an approved version: kinematic URDF plus part-local clouds and evidence.

A box demonstration can build the body and first lid, turn the box, recover the
same lid on its opposite face, and later discover lid2 by asking the user to hold
lid1 and move only lid2. Chain discovery is a later milestone; it is not already
implemented by the current star-root tracker. Linked joint motion alone may not
uniquely identify the chain, so independent excitation is useful.

## Object motion and representation

Use `T_camera_part(t) = T_camera_root(t) * T_root_part(q(t))`, or initially
independently measured part poses with pairwise `inv(T_camera_parent) *
T_camera_child`. Common rigid object motion cancels in that relative transform
when both poses are reliable and synchronized. A fixed camera does not imply a
fixed root. Preserve root identity through translation/rotation, with explicit
root-pose loss when occluded; do not silently promote another part.

Current `_pick_root` in `naive.py` ranks camera-frame translation spread. This
is a moving-object risk and must be tested/replaced or bypassed with an explicit
user-selected reference body. Claude owns that file during probation work;
coordinate a separate change rather than mixing it into the current ablation.

Store persistent part ID, local observed cloud, keyframes/view directions,
current pose and provenance. Keep root pose, joint hypotheses, provisional
surface fragments and user constraints separate. Gaussian appearance is an
optional rendering/refinement layer; RGB-D geometry remains the first baseline.
Do not fuse uncertain frames irreversibly: retain frame-to-part provenance or
buffer observations until pose/association checks pass, allowing re-fusion after
user corrections. Missing surfaces remain unknown, not certified free space.

## Online guidance: first implement a rule-based policy

| Evidence | Suggested action | Update sought |
| --- | --- | --- |
| Pose support lost / depth disagreement | Return toward last reliable view | Reacquisition before fusion |
| Part separation or joint type ambiguous | Hold one body and move the candidate | Relative-motion evidence |
| Hinge/slider estimate unstable | Repeat a wider comfortable motion | Better axis/type conditioning |
| Few view directions / exposed boundary | Turn object while keeping joint configuration | New observed surfaces |
| Opposite-face association ambiguous | Show an intermediate view or identify same part | Identity correspondence |
| Two joints move together | Hold intermediate link, move terminal link | Chain identifiability |

Prioritize reacquisition over accumulating geometry, then ambiguous structure,
then additional views. Use hysteresis and a cooldown so prompts do not flicker.
Initially use a small library of actions, not an unvalidated learned policy or
claims of calibrated information gain. View diversity measures observed evidence;
it cannot yield a truthful percentage of unknown total surface completeness.

## Human input and approval semantics

Provide only a few initial controls: set reference body, accept/reject proposed
part, mark two fragments as the same part, undo, and save model version. Add
joint type/parent edits only after the simpler flow is validated. Human-confirmed
topology is a constraint with recorded provenance, not proof of tracking accuracy.
Continue measuring residuals and flag contradictions. User-entered motion bounds
and observed motion ranges are separate from verified physical joint limits.

Save timestamped actions with frame IDs and model version. In replay, apply each
action only when its original frame is reached. Report user correction count and
interaction time; never compare assisted input against autonomous input as if
they were the same setting. Merely showing suggestions in a prerecorded clip
does not test active guidance: the motion must respond in simulation or capture.

## Validation and ordering

First: record RGB-D, masks and user events; persistent reference-body selection;
separate scan/exercise instructions; provisional cloud preview and save/undo.
Next: measured rule-based guidance and reprojection feedback. Later: chain
authoring, richer edits and task-specific robot contact previews.

Matched SAPIEN controls should distinguish whole-object rigid motion, articulation
only, and simultaneous motion, plus occlusion/opposite-face transitions. GT is
evaluation-only; object/part clicks are explicit user supervision. Evaluate joint
type/axis, part pose and correct ID coverage, false splits/merges, held-out-view
geometry, fusion corruption, end-to-end latency, and time/actions to a usable model.

Ablations: unguided versus scripted versus evidence-driven demonstrations;
guidance without edits versus guidance with edits; automatic root versus explicit
reference-body selection; provisional fusion versus immediate fusion; geometry
versus optional Gaussian refinement at comparable compute. Novelty is unproven.

## Established starting points

- [Pillai, Walter and Teller: Learning Articulated Motions From Visual
  Demonstration](https://arxiv.org/abs/1502.01659): human demonstration, sparse
  tracking, motion segmentation, component poses and articulation inference.
- [Sturm, Stachniss and Burgard: A Probabilistic Framework for Learning
  Kinematic Models of Articulated Objects](https://arxiv.org/abs/1405.7705):
  graph-based joint model estimation and selection from noisy part poses.
- [Structure from Action](https://sfa.cs.columbia.edu/): interaction selection,
  accumulated part discovery, joint estimation and articulated model creation.

The proposed contribution to investigate is reducing the demonstrations and
corrections needed for a verifiable model through online feedback, rather than
claiming interactive articulation learning itself is new.

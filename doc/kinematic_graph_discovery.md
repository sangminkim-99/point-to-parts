# Kinematic graph from pairwise relative motion — first measurement

Author: Claude, 2026-09-11/12. Status: **offline analysis tool + matched
measurement; nothing changed in the tracker; nothing promoted.** This is the
first step of the user-approved direction recorded on 2026-09-10 (dialogue
"relative-motion joint discovery"): joints from pairwise `T_ij(t)`, graph
selection afterwards, root only for representation.

Tool: `examples/multi_part/kinematic_graph.py`. Inputs: `replay.py --trace-out`
(live per-frame poses + GT for scoring) and `replay.py --dump-hist` (the
tracker's own per-part histories, **including retro back-fill**, plus each
part's anchor points, fit sigma and rotation floor). Outputs under
`results/kinematic_graph_v1/` (`*_h_graph.json`, stdouts, traces, hists).

## Why this was the next thing

`doc/demo_checkpoint_20260911.md` put "correct parent graph" first. On RBO
ikeasmall02 the star-root tracker reports two prismatic joints with axis error
0.5° and 2.3° — and **both parents wrong, both rows on the same GT drawer**
(`results/frozen_gate_rbo_v1/ikea_joint_identity_rerun.md`). The root (part 0)
is a drawer: `_pick_root` takes the part with the least camera translation,
and the body parts are tracked worse than that drawer.

## Method (what the tool does)

1. For every pair of persistent part ids, on the frames both have a pose,
   `A_t = inv(T_i) @ T_j`. Whole-object motion cancels by construction.
2. Fit rigid / prismatic / revolute (and optionally the free model) with the
   same `JointModel` and the same conditioning the tracker uses for its live
   joint: `sigma = max(sigma_i, sigma_j)`, `sigma_r_floor = max(floor_i,
   floor_j)`, `geom` = parent points + child points carried by the rest
   transform, axis prior 0.2 × radius, revolute offered only above 1.5°.
   Without this conditioning the fits do NOT reproduce the tracker (a drawer
   reads as a 74°-off revolute); with it they match to the digit (13.5°,
   23.8° on the two sim drawers, identical to the star joints).
3. Pairs whose best model is rigid are one body (never fired, see below).
4. Kruskal minimum spanning tree over bodies; edge weight = best BIC **per
   observation** (histories differ in length; raw BIC would just prefer short
   ones). Articulated pairs beat "disconnected" regardless of cost.
5. Root: `--root spread` (default) = the tracker's least-translation rule;
   `--root degree` = the body with the most tree neighbours, spread as
   tie-break. Both are reported; see Controls for where each fails.
6. GT only for scoring: part → mocap body by pose attachment (falls back to
   track votes on ties), edge ∈ spec joints, parent direction vs spec once
   rooted, axis angle vs a reference joint fitted to the mocap relative poses
   (forced to the spec type, because a mocap slide is also a huge revolute arc).

## Result on ikeasmall02 (RBO, SAM2 masks, 240 frames, `reprojection_split`)

Part identities (by pose): p0→rb1 (upper drawer), p1→rb2, p3→rb2 (lower
drawer, twice), p2→rb0, p4→rb0 (body, twice). 5 parts vs 3 GT, as before.

Pairwise fits (tracker-conditioned, retro histories):

| pair | GT | n | kind | cost/obs | smooth | span |
| --- | --- | --- | --- | --- | --- | --- |
| 0-2 | rb1-rb0 | 229 | prismatic | **35.3** | 0.73 | 0.099 m |
| 2-3 | rb0-rb2 | 229 | prismatic | **37.1** | 0.98 | 0.124 m |
| 0-3 | rb1-rb2 (drawer-drawer) | 239 | prismatic | 66.4 | 1.00 | 0.161 m |
| 0-1 | rb1-rb2 | 239 | prismatic | 38.2 | 1.00 | 0.160 m |
| 1-2 | rb2-rb0 | 229 | prismatic | 72.8 | 0.98 | 0.137 m |
| 1-3 | rb2-rb2 (same body) | 239 | revolute | 5.3 | 0.83 | 0.146 |
| 2-4 | rb0-rb0 (same body) | 207 | revolute | 2.3 | 0.40 | 1.214 |
| 0-4, 1-4, 3-4 | ×-rb0 (p4) | 211 | revolute | 40–76 | ~0.4 | ~1.1 |

Tree with `--root degree` (root = body p2 = rb0); with `--root spread` the
root is p0 (the upper drawer) and the p0–p2 edge reads rb0 → rb1 instead,
so parents score 1/2 rather than 2/2:

| edge | GT | spec edge | parent ok | type ok | axis vs mocap |
| --- | --- | --- | --- | --- | --- |
| p0 → p2 | rb1 → rb0 | yes | **yes** | yes | 39.2° |
| p3 → p2 | rb2 → rb0 | yes | **yes** | yes | 21.3° |
| p4 → p2 | rb0 → rb0 | no (duplicate body) | — | — | — |
| p1 → p3 | rb2 → rb2 | no (duplicate body) | — | — | — |

**Star-root (current tracker): 2 joint rows, 1 unique GT joint, 1 duplicate,
parents 0/2 correct. Pairwise tree: 2 edges that are spec joints, 2 unique
GT joints, parents 2/2 correct, types 2/2.** The chain ambiguity resolves the
way it should: the drawer–drawer pair (0-3) costs 66 per observation, the two
body–drawer pairs cost 35 and 37, because the two drawers move at different
times and their relative motion is not one degree of freedom.

Three honest qualifications:

- **Axis error got worse, not better** (0.5°/2.3° → 39°/21°). The star axes
  were measured against the wrong parent (a drawer that happened to be still),
  which is why they looked exact. Against the right parent the axis inherits
  the body parts' tracking error: p2 drifts 18→37 mm over its life, p4 (born
  f209) has a retro history with 0.8 m of spread. Correct topology exposes the
  body-tracking problem instead of hiding it.
- **Root by spread still picks the drawer** (p0 spread 0.074 m < p2 0.103 m).
  With that root one edge flips (rb0 → rb1) and parents score 1/2. The
  degree rule is a representation choice, stated as such; it is right here
  because a body is the hub of a cabinet, not because it is still.
- **Rigid merges never fire.** The same-body pairs (1-3, 2-4) are typed
  revolute with tiny cost, not rigid: their relative motion is real (p1 sits
  112 mm off rb2 for its whole life; p4's retro history is garbage), so no
  pairwise test can honestly call them one body. The 5-vs-3 over-segmentation
  is a tracking/identity problem upstream of the graph.

## Controls (matched: same tool, tracker-conditioned fits from `--dump-hist`)

| case | tracker parts / GT | star-root: parent ok, axis | pairwise tree (root=spread): parent ok, axis | root=degree |
| --- | --- | --- | --- | --- |
| laptop_hinge (sim) | 2 / 2 | 1/1, 9.6° | 1/1, 8.9° | same |
| laptop_hinge_noisy (sim, 3 mm noise) | 2 / 2 | 1/1, 3.9° | 1/1, 3.7° | same |
| storage_slide (sim) | 2 / 2 | 1/1, 4.7° | 1/1, 3.3° | same |
| laptop_orbit (sim, rigid, 360° camera orbit) | 1 / 1 | no joint (correct) | 0 edges (correct) | same |
| laptop_hinge_orbit (sim) | 2 / 2 | no joint row | unscorable: both parts map to `base` (p0 pose error 1 m / 178° vs GT) | — |
| cardboardbox01 (RBO, fallback gate) | 3 / 2 | 1/1, 3.9° | **1/1, 4.2°** + duplicate body edge | **0/2** — lid becomes the hub |
| storage_door_drawer (sim, staggered) | 2 / 3 | 1/1, 13.5° (drawer only) | 1/1, 13.5° | same |
| storage_two_drawers_v2 (sim, staggered) | 2 / 3 | 1/1, 23.8° (one drawer only) | 1/1, 23.8° | same |
| **ikeasmall02 (RBO)** | 5 / 3 | **0/2**, 1 unique joint, 1 duplicate | **1/2**, 2 unique joints | **2/2**, 21°/39° |

Where the tracker found the right parts, the pairwise tree reproduces the
star joint exactly (same parent, axis within 1° — the same observations and
the same estimator). It changes the answer only where the star root was wrong
(ikea) — and there the answer depends on the root rule:

- `--root spread` (least camera translation, the tracker's `_pick_root`):
  ikea 1/2 (the drawer p0 is still the root, one edge flips), cardboard 1/1.
- `--root degree` (most tree neighbours): ikea 2/2, **cardboard 0/2** — the two
  body fragments p0 and p2 both attach to the lid p3 (their own relative
  motion, 20° of spurious rotation, costs 32 per observation vs 11–12 for
  lid–body), so the lid has degree 2 and becomes the root.

Neither rule is safe while duplicate bodies exist; the tool defaults to
`spread` (no new claim) and reports both. The user's framing stands: the root
is a representation choice, and the graph's *undirected* edges are right in
both RBO cases (ikea: both spec joints found once each, drawer–drawer edge
rejected; cardboard: lid–body found, 4.2°). What the direction needs is not a
better root heuristic but fewer duplicate bodies.

Two staggered chain-ambiguity controls were rendered for this
(`scripts/sim/render_partnet_sequence.py --stagger`, new): PartNet 40147
door+drawer and 40417 two drawers (0.6 m, side view, 126 mm travel each), each
joint moving in its own sub-window. **The tracker split only one of the two
children in each** (door_drawer: the drawer at f129 under `frozen_fallback`,
never the 41° door; two_drawers: the second drawer at f135, never the first).
Refusals were dominated by "the disagreeing points had not been seen together
often enough to group" (`co_min_seen`; 133 and 202 refusals) and by the
contradiction gate (gain ≈ 0.001, 71 rejections on the door). So these
controls could not exercise the drawer–drawer ambiguity in the tracker; the
ambiguity was exercised only on real ikea, where it resolved correctly.
The staggered renders are a split-discovery negative result in their own
right (`results/kinematic_graph_v1/storage_*_render.log`, `*.stdout`).

## What the trace alone cannot do

The first version used only the live trace. It reproduced neither the
tracker's types nor its axes: a part split after its joint stopped moving
(door_drawer p1 at f129, drawer stopped at f117) has **no motion at all** in
the live trace, while the tracker's joint is built from retro back-fill. Any
offline or in-tracker graph selection has to use `NaivePart.hist`, not the
per-frame poses — which is why `--dump-hist` exists and why the in-tracker
version (next step) will run in `_reparent` on `hist`.

## Recommendation

- Implement `joint_graph` (default off) in `NaivePartTracker._reparent`:
  pairwise fits on `hist` with the live conditioning, MST over parts, root by
  the existing `_pick_root` (spread) so the only change is the tree; set
  `p.parent` from the tree and rebuild each joint against that parent. The
  evaluator's `parent_match` then scores it directly. Star-root stays default.
- The duplicate-body problem (ikea p1/p3, p2/p4; cardboard p0/p2) is what
  breaks both root rules and blocks rigid merging; it is the upstream fix.
- Do not expect axis numbers to improve from this alone; the body parts'
  tracking is the bottleneck now visible (ikea p2/p4). Track that separately.
- Keep `--root spread` reportable: it is the failure mode the user described
  (a moving/badly tracked "stillest" part), and it is measurable per sequence.

# Split discovery under staggered motion — why one child was missed, and a default-off fix

Author: Claude, 2026-09-12. Status: **diagnosed with per-proposal refusal
logs; one opt-in option (`split_gain_on_child` + `split_gain_child_min_over`)
measured on 7 cases; nothing promoted.** Artifacts:
`results/split_purity_v1/*gainchild*`, `door_drawer_fb_debug*`.

## Symptom

The two staggered PartNet controls (`--stagger`, door+drawer 40147 and two
drawers 40417, each child moving in its own sub-window) always ended with
**2 parts vs 3 GT**: the tracker split only the second mover (the drawer at
f129, the second drawer at f135), never the first (door f36–63, first drawer
f48–84). `doc/kinematic_graph_discovery.md` recorded this as open.

## Diagnosis (door+drawer, `frozen_fallback` gate, `NAIVE_DEBUG` + refusal log)

The trigger works: during the door phase `out_pts` rises 15→37 and `over`
climbs 3→63, so `_split` is called every frame from f40 on. Every proposal is
refused, by two mechanisms:

1. **Co-association is too slow for a small part.** `_cluster_co` requires the
   median pairwise co-visibility weight `seen ≥ co_min_seen` (2.5). The weight
   added per frame is scaled by how decisive the frame is and by how much of
   the part disagrees; with the door at ~13 % of the points it accumulates
   ≈0.1 per frame. `seen` reaches 0.3 by f48, 0.9 by f64, 2.1 by f92 and 2.5
   only at f108 — the door stopped moving at f63.
2. **The frame path's gain test cannot see a small child.** From f64 the
   single-frame RANSAC finds the door: a 19–20-point child 9–10σ from the
   body. It is refused 44 frames running as `no_gain`: the test compares the
   MEDIAN residual over ALL points before and after the split, and 20 points
   out of 270 cannot move a median that 250 body points at 1 mm set. (Until
   f60 the same proposals were `too_small`: 7.6° / 5 mm, under the joint-size
   thresholds, and `one_motion` — correct refusals while the door had barely
   opened.)

## Option (default off)

`split_gain_on_child`: read the gain on the points that change hands — the
smaller group's residual under the one-body pose against its residual under
its own motion. `split_gain_child_min_over` (30): only once the differential
motion has persisted that many frames, because read on the child alone the
test also accepted a 22-point laptop lid one frame earlier than the pooled
test, and that smaller child's pose later flipped (1.1 m error, purity
96.6 → 83.1 %). With the persistence requirement that run is identical to
base; the door (over 23–63 when refused) still passes.

## Matched results (`split_gain_on_child=true`, min_over 30; everything else as the case's base)

| case | base | gain on child (min_over 30) | gain on child, no min_over |
| --- | --- | --- | --- |
| storage_door_drawer (fb) | 2/3, drawer only, purity 74 % | **3/3, 94.9 %**, door revolute 11.5° / 0.4 cm, drawer 1.4–7.2°, parents 2/2, door pose 2.1 mm | 3/3, 95.0 % |
| storage_two_drawers_v2 | 2/3, one drawer, 80.7 % | **3/3, 99.0 %**, prismatic 4.0° / 10.1°, parents 2/2 | same |
| laptop_hinge | 2/2, 96.6 %, lid 9.6° | identical | 2/2 but 83.1 %, lid pose 1120 mm / 145° |
| storage_slide, laptop_orbit | 2/2 · 1/1 | identical | identical |
| ikeasmall02 (residual_veto + refine) | 3/3, 94.0 % | identical | 3/3 but 92.8 %, rb2 axis 1.7 → 7.5° |
| cardboardbox01 (residual_veto + refine) | 2/2, 96.8 % | identical | identical |

The staggered chain-ambiguity controls are now fully exercised in the
tracker: both children found, both joints typed and attached correctly under
the star root (the sim base is part 0). The co-association slowness (1) is
untouched — it is a weighting question (`w` shrinks with the part's share of
the points) and a threshold (`co_min_seen`), and changing either affects
every split decision; measured but not attempted here.

## Not established

- Real captures with a brief small-part motion (the RealSense clips have no
  GT); other RBO objects.
- Whether `min_over = 30` (1 s at 30 fps) is the right persistence; it was
  chosen as the smallest round number above the laptop lid's ~15 and below
  the door's 23 and then confirmed, not swept.

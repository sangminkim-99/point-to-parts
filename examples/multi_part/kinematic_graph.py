#!/usr/bin/env python
"""Kinematic graph from pairwise relative part motion -- no reference body.

Offline analysis over a saved pose trace (`replay.py --trace-out`). The current
tracker builds a STAR: every part's joint is measured against one root chosen
by least camera-frame translation. On ikeasmall02 that root is a drawer, so
every joint's parent is wrong even where the axis is exact. This tool asks the
question the way the user asked it to be asked:

    for every pair (i, j) of persistent part ids, on the frames both were
    observed, T_ij(t) = inv(T_cam_i(t)) @ T_cam_j(t); shared whole-object
    motion cancels. Fit rigid / prismatic / revolute / disconnected on each
    history (JointModel, Sturm-style BIC), then choose the graph:
      1. pairs whose relative motion is best explained as RIGID are one body;
      2. a minimum-cost spanning tree over the bodies is the kinematic tree;
      3. a root is chosen only afterwards, for the tree/URDF representation.

GT (RBO mocap + spec) is used ONLY to score the result: which mocap body each
part id is (by track votes, as the evaluator does), whether each tree edge is a
spec joint, whether its parent direction matches the spec once rooted, and the
axis angle against a reference joint fitted to the mocap relative poses.

Usage:
  python -m examples.multi_part.kinematic_graph results/.../trace.npz \
      --seq-dir /path/to/RBO/sequences/ikeasmall02_o [--out report.json]
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from point2pose.pipeline.components.joint_model import JointModel


# ---------------------------------------------------------------- loading --

def load_trace(path):
    d = np.load(path, allow_pickle=True)
    return {"frames": d["frames"].astype(int),
            "ids": [int(x) for x in d["part_ids"]],
            "poses": d["poses"],                    # (F, N, 4, 4), NaN = absent
            "votes": d["votes"],                    # (F, N, G)
            "gt_names": [str(x) for x in d["gt_names"]],
            "gt_poses": d["gt_poses"]}              # (F, G, 4, 4)


def apply_hist(tr, path, gate=0.02):
    """Replace the live per-frame poses with the tracker's own part histories
    (`replay.py --dump-hist`), which include the retro back-fill a newborn part
    receives for frames before its birth. The live trace only has poses from
    the birth frame on, so a part split after its joint stopped moving carries
    no motion at all there -- the tracker's joint does. Frames the history
    marks with a residual at or above `gate` are dropped as the tracker does."""
    d = np.load(path, allow_pickle=True)
    poses = np.full_like(tr["poses"], np.nan)
    tix = {int(f): t for t, f in enumerate(tr["frames"])}
    n_before = {}
    tr["part_fit"] = {}
    for pid in [int(x) for x in d["part_ids"]]:
        if pid not in tr["ids"]:
            continue
        k = tr["ids"].index(pid)
        fr, P = d[f"frames_{pid}"], d[f"poses_{pid}"]
        res = d[f"resid_{pid}"] if f"resid_{pid}" in d else np.full(len(fr), np.nan)
        for f, T, r in zip(fr, P, res):
            t = tix.get(int(f))
            if t is not None and (np.isnan(r) or r < gate):   # _rebuild_joint's gate
                poses[t, k] = T
        if f"points_{pid}" in d:
            tr["part_fit"][pid] = {"points": d[f"points_{pid}"],
                                   "sigma": float(d[f"sigma_{pid}"]),
                                   "rot_floor": float(d[f"rot_floor_{pid}"])}
        live_first = np.flatnonzero(np.isfinite(tr["poses"][:, k, 0, 0]))
        first = int(tr["frames"][live_first[0]]) if live_first.size else None
        n_before[pid] = int((fr < first).sum()) if first is not None else 0
    tr["poses"] = poses
    return n_before


def gt_identity(tr):
    """part_id -> mocap body name, by summed track votes (evaluator convention).

    Returns None for a part whose tracks were never labelled."""
    out = {}
    tot = tr["votes"].sum(0)                        # (N, G)
    for k, pid in enumerate(tr["ids"]):
        if tot[k].sum() == 0:
            out[pid] = None
        else:
            out[pid] = tr["gt_names"][int(np.argmax(tot[k]))]
    return out


def finite_pose(T):
    return T is not None and np.all(np.isfinite(T))


def gt_identity_by_pose(tr, min_frames=5):
    """part_id -> the mocap body it stays rigidly attached to.

    Evaluation only. For each GT body g, inv(T_gt_g(t)) @ T_part(t) should be
    constant if the part IS that body; the body with the smallest translation
    spread of that relative pose wins. Track votes (the evaluator's rule) label
    where a track was BORN, which on a camera-orbit sequence labelled a lid
    part 'base'; the pose criterion is what the graph is about, so it is the
    fairer identity for scoring the graph. Returns (name, spread_m) or
    (None, None) when the part has too few frames with GT."""
    out = {}
    for k, pid in enumerate(tr["ids"]):
        best = (None, None)
        spreads = {}
        for g, name in enumerate(tr["gt_names"]):
            rel = []
            for t in range(len(tr["frames"])):
                Tp, Tg = tr["poses"][t, k], tr["gt_poses"][t, g]
                if finite_pose(Tp) and finite_pose(Tg):
                    rel.append((np.linalg.inv(Tg) @ Tp)[:3, 3])
            if len(rel) < min_frames:
                continue
            rel = np.array(rel)
            spread = float(np.linalg.norm(rel.max(0) - rel.min(0)))
            spreads[name] = spread
            if best[1] is None or spread < best[1]:
                best = (name, spread)
        # a part that did not move relative to ANY body (born after its joint
        # stopped) ties everywhere; the pose says nothing, so decline
        if len(spreads) >= 2:
            srt = sorted(spreads.values())
            if srt[0] > 0.8 * srt[1]:
                best = (None, best[1])
        out[pid] = best
    return out


# --------------------------------------------------------- pairwise fits --

def relative_history(tr, a, b):
    """(frames, [inv(T_a) @ T_b]) over frames where both parts were tracked."""
    ka, kb = tr["ids"].index(a), tr["ids"].index(b)
    fr, rel = [], []
    for t, f in enumerate(tr["frames"]):
        Ta, Tb = tr["poses"][t, ka], tr["poses"][t, kb]
        if finite_pose(Ta) and finite_pose(Tb):
            fr.append(int(f))
            rel.append(np.linalg.inv(Ta) @ Tb)
    return fr, rel


def fit_pair(rel, min_obs, sigma, radius, allow_free, axis_prior, revolute_min_deg,
             fit_a=None, fit_b=None):
    """Fit the hypotheses on one relative history the way the tracker does
    (`NaivePartTracker._new_joint` + the per-frame conditioning in `step`):
    free model off by default, axis-location prior scaled by the parts'
    geometry, noise = max of the two parts' fit sigmas, rotation floor = max of
    their lever-arm floors. With `--hist` the real anchor points, sigma and
    floor of each part are available (fit_a / fit_b, a = parent side) and
    `_joint_geom` is reproduced: parent points plus child points carried into
    the parent frame by the rest transform A0. Without them the trace carries
    no geometry, so `radius` stands in (a cube of that half-size) and the
    default sigma is used. None if the history is too short."""
    if len(rel) < min_obs:
        return None
    jm = JointModel(min_obs=min_obs)
    jm.allow_free = allow_free
    jm.axis_prior = axis_prior
    jm.axis_max = 0.0
    jm.min_angle = np.radians(revolute_min_deg)
    jm.sigma = sigma
    if fit_a is not None and fit_b is not None:
        jm.sigma = max(fit_a["sigma"], fit_b["sigma"])
        jm.sigma_r_floor = max(fit_a["rot_floor"], fit_b["rot_floor"])
        A0 = rel[0]
        b = fit_b["points"] @ A0[:3, :3].T + A0[:3, 3]
        jm.geom = np.concatenate([fit_a["points"], b]).astype(np.float64)
    elif radius > 0:
        c = np.array(list(itertools.product([-radius, radius], repeat=3)))
        jm.geom = np.vstack([c, np.zeros((1, 3))])
    for A in rel:
        jm.add(A)
    if not jm.fit():
        return None
    c = jm.confidence()
    n = len(jm.A)
    return {"model": jm, "n": n, "kind": jm.kind,
            "bic": {k: float(v) for k, v in jm.bic.items()},
            # per-observation cost, so histories of different length compare
            "cost": float(min(jm.bic.values())) / n,
            "rmse_sigma": float(c.get("rmse", float("nan"))),
            "sigma_t_mm": 1000 * float(c.get("sigma_t", float("nan"))),
            "smooth": float(c.get("smooth", 0.0)),
            "type_p": float(c.get("type_p", 0.0)),
            "axis_rms_deg": float(c.get("axis_rms_deg", float("nan"))),
            "span": float(c.get("span", 0.0))}


def pairwise(tr, min_obs, fit_kw):
    out = {}
    for a, b in itertools.combinations(tr["ids"], 2):
        fr, rel = relative_history(tr, a, b)
        pf = tr.get("part_fit", {})
        fit = fit_pair(rel, min_obs, fit_a=pf.get(a), fit_b=pf.get(b), **fit_kw)
        if fit is None:
            out[(a, b)] = {"n": len(rel), "kind": None, "frames": fr}
            continue
        fit["frames"] = fr
        out[(a, b)] = fit
    return out


# --------------------------------------------------------- graph choice --

class _UF:
    def __init__(self, xs):
        self.p = {x: x for x in xs}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def merge_rigid(ids, pairs, rigid_margin):
    """Union parts whose relative motion is best explained by a rigid link.

    `rigid_margin` is how much lower (per observation) the rigid BIC must be
    than the best articulated alternative; 0 = any rigid win counts."""
    uf = _UF(ids)
    merged = []
    for (a, b), f in pairs.items():
        if f.get("kind") != "rigid":
            continue
        alt = [v for k, v in f["bic"].items() if k in ("prismatic", "revolute")]
        gap = (min(alt) - f["bic"]["rigid"]) / f["n"] if alt else float("inf")
        if gap >= rigid_margin:
            uf.union(a, b)
            merged.append({"pair": [a, b], "n": f["n"], "rigid_gap_per_obs": gap})
    bodies = {}
    for x in ids:
        bodies.setdefault(uf.find(x), []).append(x)
    return list(bodies.values()), merged


def spanning_tree(bodies, pairs):
    """Kruskal over bodies; edge weight = lowest per-observation cost among the
    articulated pairs joining the two bodies. Disconnected-typed pairs are used
    only if nothing else connects a body (reported as such)."""
    body_of = {x: i for i, bs in enumerate(bodies) for x in bs}
    cand = {}
    for (a, b), f in pairs.items():
        if f.get("kind") in (None, "rigid"):
            continue
        i, j = body_of[a], body_of[b]
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        # articulated beats disconnected regardless of cost
        rank = (0 if f["kind"] in ("prismatic", "revolute") else 1, f["cost"])
        if key not in cand or rank < cand[key]["rank"]:
            cand[key] = {"rank": rank, "pair": (a, b), "fit": f}
    uf = _UF(range(len(bodies)))
    edges = []
    for key, c in sorted(cand.items(), key=lambda kv: kv[1]["rank"]):
        i, j = key
        if uf.find(i) != uf.find(j):
            uf.union(i, j)
            edges.append({"bodies": [i, j], "pair": list(c["pair"]), "fit": c["fit"]})
    components = len({uf.find(i) for i in range(len(bodies))})
    return edges, components


def pick_root(tr, bodies, ids_by_body):
    """Representation root: the body whose origin visited the smallest box on
    the frames every body shares. A coordinate convention, not a claim that the
    body is stationary."""
    common = None
    for bs in bodies:
        f = set()
        for x in bs:
            k = tr["ids"].index(x)
            f |= {int(tr["frames"][t]) for t in range(len(tr["frames"]))
                  if finite_pose(tr["poses"][t, k])}
        common = f if common is None else common & f
    spread = []
    for bs in bodies:
        # the body's representative id: the one with the most observations
        rep = max(bs, key=lambda x: np.isfinite(tr["poses"][:, tr["ids"].index(x), 0, 0]).sum())
        k = tr["ids"].index(rep)
        pts = [tr["poses"][t, k][:3, 3] for t in range(len(tr["frames"]))
               if int(tr["frames"][t]) in (common or set()) and finite_pose(tr["poses"][t, k])]
        pts = np.array(pts)
        spread.append(float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else float("inf"))
    return int(np.argmin(spread)), spread, sorted(common or [])


def pick_root_by_degree(edges, n_bodies, spread):
    """Representation root = the body most others articulate against (highest
    tree degree), spread as the tie-break. A cabinet body carries every drawer
    and door; a laptop base carries its lid. This does not need the root to be
    the stillest thing in the camera, which under whole-object motion or poor
    body tracking it is not."""
    deg = [0] * n_bodies
    for e in edges:
        i, j = e["bodies"]
        deg[i] += 1; deg[j] += 1
    return int(min(range(n_bodies), key=lambda i: (-deg[i], spread[i]))), deg


def root_tree(edges, n_bodies, root):
    """Orient the undirected tree away from the root: body -> parent body."""
    adj = {i: [] for i in range(n_bodies)}
    for e in edges:
        i, j = e["bodies"]
        adj[i].append(j); adj[j].append(i)
    parent = {root: None}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                stack.append(v)
    return parent


# ------------------------------------------------------------ GT scoring --

def gt_spec(seq_dir):
    """Spec joints (child -> {parent, type}) and base part, RBO or SAPIEN sim.

    Both readers normalise joint types to "prismatic" / "revolute" and expose
    `base_part`; the spec is the evaluation reference only."""
    from pathlib import Path
    if (Path(seq_dir) / "camera_rgb").is_dir():
        from point2pose.io.sources.dataset.rbo_reader import RBOReader
        r = RBOReader(seq_dir)
    else:
        from point2pose.io.sources.dataset.sapien_reader import SapienReader
        r = SapienReader(seq_dir)
    return r, {j["child"]: j for j in r.joints}, r.base_part


def axis_error_deg(tr, fit, a, b, gt_a, gt_b, spec_type=None):
    """Median camera-frame angle between our axis (parent frame a) and a
    reference joint fitted to the mocap relative poses of the same body pair."""
    ka = tr["ids"].index(a)
    ga, gb = tr["gt_names"].index(gt_a), tr["gt_names"].index(gt_b)
    ref = JointModel()
    if spec_type == "prismatic":
        ref.min_angle = np.pi          # never offer the revolute reading of a slide
    for t in range(len(tr["frames"])):
        Ta, Tb = tr["gt_poses"][t, ga], tr["gt_poses"][t, gb]
        if finite_pose(Ta) and finite_pose(Tb):
            ref.add(np.linalg.inv(Ta) @ Tb)
    if not ref.fit() or ref.axis is None or fit["model"].axis is None:
        return None, ref.kind
    angs = []
    for t, f in enumerate(tr["frames"]):
        if int(f) not in fit["frames"]:
            continue
        Oa, Ta = tr["poses"][t, ka], tr["gt_poses"][t, ga]
        if not (finite_pose(Oa) and finite_pose(Ta)):
            continue
        u = Oa[:3, :3] @ fit["model"].axis
        v = Ta[:3, :3] @ ref.axis
        angs.append(float(np.degrees(np.arccos(np.clip(
            abs(u @ v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-9), -1, 1)))))
    return (float(np.median(angs)) if angs else None), ref.kind


# ------------------------------------------------------------------ main --

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--seq-dir", default=None, help="RBO sequence dir, for the spec (GT scoring)")
    ap.add_argument("--min-obs", type=int, default=12, help="shared frames a pair needs to be fitted")
    ap.add_argument("--sigma", type=float, default=0.003, help="metres, JointModel noise floor")
    ap.add_argument("--object-radius", type=float, default=0.25,
                    help="metres; stands in for the part geometry the axis prior is scaled by")
    ap.add_argument("--allow-free", action="store_true", help="admit the 6-DoF disconnected model (tracker default: off)")
    ap.add_argument("--axis-prior", type=float, default=0.2, help="tracker default 0.2 x radius")
    ap.add_argument("--revolute-min-deg", type=float, default=None, help="tracker default from NaiveConfig")
    ap.add_argument("--rigid-margin", type=float, default=0.0,
                    help="per-obs BIC margin rigid must win by to merge two parts")
    ap.add_argument("--hist", default=None,
                    help="replay --dump-hist npz: use the tracker's part histories "
                         "(with retro back-fill) instead of the trace's live poses")
    ap.add_argument("--root", choices=["spread", "degree"], default="spread",
                    help="representation root: the body with the least camera-frame translation (the tracker's rule, default) "
                         "or the one with the most tree neighbours; both fail in the presence of duplicate bodies, see doc")
    ap.add_argument("--identity", choices=["pose", "votes"], default="pose",
                    help="how a part id is mapped to a GT body for scoring (evaluation only)")
    ap.add_argument("--out", default=None, help="json report")
    args = ap.parse_args()

    tr = load_trace(args.trace)
    ids = tr["ids"]
    by_votes = gt_identity(tr)              # from the live trace, before any hist swap
    if args.hist:
        n_before = apply_hist(tr, args.hist)
        print(f"[graph] using tracker histories from {args.hist}; retro frames before "
              f"live birth per part: {n_before}")
    by_pose = gt_identity_by_pose(tr)
    ident = ({pid: (by_pose[pid][0] if by_pose[pid][0] is not None else by_votes[pid]) for pid in ids}
             if args.identity == "pose" else by_votes)
    print(f"[graph] {len(ids)} part ids over {len(tr['frames'])} frames")
    print(f"[graph] gt identity by track votes: {by_votes}")
    print(f"[graph] gt identity by pose (body, rel. spread m): "
          f"{ {p: (v[0], round(v[1], 3) if v[1] is not None else None) for p, v in by_pose.items()} }")
    print(f"[graph] scoring with identity = {args.identity}: {ident}")

    if args.revolute_min_deg is None:
        from examples.multi_part.naive import NaiveConfig
        args.revolute_min_deg = NaiveConfig().revolute_min_deg
    fit_kw = dict(sigma=args.sigma, radius=args.object_radius, allow_free=args.allow_free,
                  axis_prior=args.axis_prior, revolute_min_deg=args.revolute_min_deg)
    print(f"[graph] fit settings: {fit_kw}")
    pairs = pairwise(tr, args.min_obs, fit_kw)
    print("\n[graph] pairwise relative-motion fits (cost = best BIC per observation):")
    print(f"  {'pair':>9} {'gt':>10} {'n':>4} {'kind':>13} {'cost':>8} {'rmse/s':>7} {'sig_mm':>6} {'smooth':>6} "
          f"{'type_p':>6} {'span':>7}  bic(rigid/prism/rev/free)")
    for (a, b), f in pairs.items():
        g = f"{ident[a]}-{ident[b]}"
        if f.get("kind") is None:
            print(f"  {a:>4}-{b:<4} {g:>10} {f['n']:>4} {'(too few)':>13}")
            continue
        bic = f["bic"]
        bs = "/".join(f"{bic.get(k, float('nan')):.0f}" for k in ("rigid", "prismatic", "revolute", "disconnected"))
        print(f"  {a:>4}-{b:<4} {g:>10} {f['n']:>4} {f['kind']:>13} {f['cost']:>8.2f} "
              f"{f['rmse_sigma']:>7.2f} {f['sigma_t_mm']:>6.2f} "
              f"{f['smooth']:>6.2f} {f['type_p']:>6.2f} {f['span']:>7.3f}  {bs}")

    bodies, merged = merge_rigid(ids, pairs, args.rigid_margin)
    print(f"\n[graph] rigid merges: {merged}")
    print(f"[graph] bodies: {bodies}  (gt: {[sorted({str(ident[x]) for x in bs}) for bs in bodies]})")

    edges, ncomp = spanning_tree(bodies, pairs)
    root_spread, spread, common = pick_root(tr, bodies, None)
    root_deg, deg = pick_root_by_degree(edges, len(bodies), spread)
    root = root_deg if args.root == "degree" else root_spread
    parent = root_tree(edges, len(bodies), root)
    print(f"[graph] spanning tree: {len(edges)} edges, {ncomp} component(s)")
    print(f"[graph] root by least camera spread: body {root_spread} {bodies[root_spread]} "
          f"(spread m {[round(s, 3) for s in spread]} over {len(common)} shared frames); "
          f"by tree degree: body {root_deg} {bodies[root_deg]} (degrees {deg}); using --root {args.root}")

    # ---- GT scoring
    report = {"ids": ids, "gt_identity": ident, "bodies": bodies, "rigid_merges": merged,
              "root_body": root, "root_rule": args.root, "root_by_spread": root_spread,
              "root_by_degree": root_deg, "body_spread_m": spread, "body_degree": deg,
              "pairs": [{"pair": [a, b], "gt": [ident[a], ident[b]], "n": f.get("n"),
                         "kind": f.get("kind"), "cost": f.get("cost"), "bic": f.get("bic"),
                         "smooth": f.get("smooth"), "span": f.get("span")}
                        for (a, b), f in pairs.items()],
              "edges": []}
    spec = base = reader = None
    if args.seq_dir:
        reader, spec, base = gt_spec(args.seq_dir)
        print(f"[graph] spec joints: {[(c, j['parent'], j['type']) for c, j in spec.items()]}, base {base}")
    print("\n[graph] tree edges (child body -> parent body after rooting):")
    n_edge_ok = n_parent_ok = 0
    for e in edges:
        i, j = e["bodies"]
        child, par = (i, j) if parent.get(i) == j else (j, i)
        a, b = e["pair"]
        f = e["fit"]
        gt_child = {str(ident[x]) for x in bodies[child]}
        gt_par = {str(ident[x]) for x in bodies[par]}
        row = {"child_body": child, "parent_body": par, "child_ids": bodies[child],
               "parent_ids": bodies[par], "fit_pair": [a, b], "kind": f["kind"],
               "n": f["n"], "cost": f["cost"], "gt_child": sorted(gt_child),
               "gt_parent": sorted(gt_par)}
        line = (f"  body{child}{bodies[child]} -> body{par}{bodies[par]}  {f['kind']:>9} "
                f"n={f['n']:>3} cost={f['cost']:.2f}  gt {sorted(gt_child)}->{sorted(gt_par)}")
        if spec is not None and len(gt_child) == 1 and len(gt_par) == 1:
            gc, gp = next(iter(gt_child)), next(iter(gt_par))
            # undirected: is {gc, gp} a spec joint at all?
            edge_ok = (gc in spec and spec[gc]["parent"] == gp) or \
                      (gp in spec and spec[gp]["parent"] == gc)
            parent_ok = gc in spec and spec[gc]["parent"] == gp
            type_ok = parent_ok and spec[gc]["type"] == f["kind"]
            ang, ref_kind = (axis_error_deg(tr, f, a if ident[a] == gp else b,
                                            b if ident[a] == gp else a, gp, gc,
                                            spec_type=spec[gc]["type"])
                             if parent_ok else (None, None))
            n_edge_ok += edge_ok; n_parent_ok += parent_ok
            row.update({"edge_is_spec_joint": bool(edge_ok), "parent_matches_spec": bool(parent_ok),
                        "type_matches_spec": bool(type_ok), "axis_err_deg": ang, "ref_kind": ref_kind})
            line += (f"  spec_edge={edge_ok} parent_ok={parent_ok} type_ok={type_ok}"
                     + (f" axis={ang:.1f}deg (ref {ref_kind})" if ang is not None else ""))
        elif spec is not None:
            row.update({"edge_is_spec_joint": None, "note": "a body maps to several/no GT bodies"})
            line += "  (body spans several GT bodies: unscorable)"
        report["edges"].append(row)
        print(line)
    if spec is not None:
        print(f"\n[graph] summary: {len(edges)} edges; spec joints {len(spec)}; "
              f"edges that are spec joints {n_edge_ok}; parent direction correct {n_parent_ok}; "
              f"bodies {len(bodies)} vs GT {len(tr['gt_names'])}")
        report["summary"] = {"edges": len(edges), "spec_joints": len(spec),
                             "spec_edges": n_edge_ok, "parent_correct": n_parent_ok,
                             "bodies": len(bodies), "gt_bodies": len(tr["gt_names"]),
                             "components": ncomp}
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1, default=lambda o: None, allow_nan=True)
        print(f"[graph] report -> {args.out}")


if __name__ == "__main__":
    main()

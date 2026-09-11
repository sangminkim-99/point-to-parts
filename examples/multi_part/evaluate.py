"""Score a streaming run against a dataset's per-part ground truth.

Mirrors run_gaussian_part_discovery.py's metrics -- gaussians are labelled from
the anchor frame's part-index map, each discovered part takes the GT part it is
mostly made of, and pose error is measured relative to the anchor.
"""
import numpy as np
from scipy.spatial.transform import Rotation as _R


def gt_labels(reader, anchor, depth0, mask0, stride):
    """GT part index per gaussian, in the order GaussianCloud.from_depth builds them."""
    pim = reader.render_part_index_map(anchor)[::stride, ::stride]
    dd, mm = depth0[::stride, ::stride], mask0[::stride, ::stride]
    return pim[(dd > 0.05) & (mm > 0)]


def gt_labels_at(reader, anchor, pts2d):
    """GT part index at each 2D query point -- the sparse counterpart."""
    pim = reader.render_part_index_map(anchor)
    u = np.clip(np.round(pts2d[:, 0]).astype(int), 0, pim.shape[1] - 1)
    v = np.clip(np.round(pts2d[:, 1]).astype(int), 0, pim.shape[0] - 1)
    return pim[v, u]


def aligned_trajectory_errors(reader, rows, pose_log, id_log, final_ids):
    """Pose-trajectory errors after one fixed per-part frame alignment.

    A newly discovered back face may use a different canonical frame from the
    original camera. Align once at its first estimated pose, never each frame.
    Persistent IDs prevent a later split from mixing another part's trajectory.
    This measures subsequent tracking, not absolute initial pose accuracy.
    """
    output = []
    for row in rows:
        identity = final_ids[row["part"]]
        reference = None
        translations, rotations = [], []
        for (frame, poses), ids in zip(pose_log, id_log):
            if identity not in ids:
                continue
            estimate = poses[ids.index(identity)]
            truth = reader.get_gt_pose(frame, row["gt"])
            if estimate is None or truth is None or not np.isfinite(estimate).all() \
                    or not np.isfinite(truth).all():
                continue
            if reference is None:
                reference = np.linalg.inv(truth) @ estimate
            expected = truth @ reference
            error = np.linalg.inv(expected) @ estimate
            translations.append(float(np.linalg.norm(error[:3, 3])) * 1000)
            rotations.append(float(np.degrees(_R.from_matrix(error[:3, :3]).magnitude())))
        if translations:
            output.append({"part_id": int(identity), "gt": row["gt"],
                           "frames": len(translations),
                           "translation_mm": float(np.median(translations)),
                           "rotation_deg": float(np.median(rotations))})
    return output


def report(reader, anchor, parts_gt, lab_gt, stream, pose_log, min_group=30):
    """Purity, coverage and per-part pose error. `pose_log` is [(frame, [T...])]."""
    out = {}
    rows = []
    for j, p in enumerate(stream.parts):
        if hasattr(p, "weights"):
            w = p.weights[:len(lab_gt)] > 0.5
        else:                            # sparse parts own indices, not weights
            w = np.zeros(len(lab_gt), bool)
            w[p.idx[p.idx < len(lab_gt)]] = True
        g = lab_gt[w[:len(lab_gt)]] if w.any() else np.array([])
        g = g[g >= 0]
        if g.size < min_group:
            continue
        cnt = np.bincount(g, minlength=len(parts_gt))
        dom = int(np.argmax(cnt))
        rows.append({"part": j, "n": int(w.sum()), "gt": parts_gt[dom],
                     "purity": float(cnt[dom] / cnt.sum())})
    out["rows"] = rows
    out["covered"] = sorted({r["gt"] for r in rows})
    out["n_gt"] = len(parts_gt)
    out["purity"] = float(np.mean([r["purity"] for r in rows])) if rows else 0.0

    Ta = {p: reader.get_gt_pose(anchor, p) for p in parts_gt}
    err = {}
    for r_ in rows:
        te, re_ = [], []
        for (i, Ts) in pose_log:
            if r_["part"] >= len(Ts) or Ts[r_["part"]] is None:
                continue
            G = reader.get_gt_pose(i, r_["gt"])
            if G is None or Ta[r_["gt"]] is None:
                continue
            Tgt = G @ np.linalg.inv(Ta[r_["gt"]])
            E = np.linalg.inv(Tgt) @ Ts[r_["part"]]
            te.append(float(np.linalg.norm(E[:3, 3])))
            re_.append(float(np.degrees(
                np.linalg.norm(_R.from_matrix(E[:3, :3]).as_rotvec()))))
        if te:
            # a GT part claimed by several discovered parts keeps the best
            cand = (float(np.median(te)) * 1000, float(np.median(re_)),
                    float(np.percentile(te, 90)) * 1000,
                    float(np.percentile(re_, 90)))
            if r_["gt"] not in err or cand[0] < err[r_["gt"]][0]:
                err[r_["gt"]] = cand
    out["err"] = err
    return out


def joint_metrics(reader, parts_gt, stream, rows, log):
    """Axis error against a joint fitted to the mocap poses, and type accuracy.

    RBO's spec declares each joint's type but not its axis, so the reference axis
    is fitted to the ground-truth relative poses with the same estimator -- the
    comparison is then against the mocap, not against our own tracking.
    Metrics follow ArtiPoint (Sturm-style): sign-agnostic axis angle in degrees
    and, for revolute, line-to-line distance.

    BOTH axes are carried into the camera frame before they are compared. Our
    axis lives in the parent part's anchor frame and the reference lives in the
    mocap rigid body's own frame, and those two frames differ by an arbitrary
    rotation, so dotting them directly measures nothing: it scored a median of
    58 deg over RBO, which is what random gives. `log` supplies our part poses
    at each evaluated frame, and the mocap pose supplies the other half.
    """
    from point2pose.pipeline.components.joint_model import JointModel
    of = {r["part"]: r["gt"] for r in rows}
    gt_type = {j["child"]: j["type"] for j in getattr(reader, "joints", [])}
    out = []
    for j, p in enumerate(stream.parts):
        if p.joint is None or p.joint.kind is None or j not in of:
            continue
        par = of.get(p.parent)
        chi = of.get(j)
        if par is None or chi is None or par == chi:
            continue
        # the base has no joint in the spec, so pairing it with anything scores
        # our estimate against a reference that does not exist
        if chi not in gt_type:
            continue
        ref = JointModel()
        for i, _ in log:
            Tp, Tc = reader.get_gt_pose(i, par), reader.get_gt_pose(i, chi)
            if Tp is None or Tc is None or not np.all(np.isfinite(Tp)) \
                    or not np.all(np.isfinite(Tc)):
                continue
            ref.add(np.linalg.inv(Tp) @ Tc)
        if not ref.fit() or ref.axis is None:
            continue
        # only frames whose part list is the final one: an index into an
        # earlier, shorter list names a different part
        angs, dists = [], []
        for i, poses in log:
            if poses is None or len(poses) != len(stream.parts):
                continue
            Op = poses[p.parent] if p.parent < len(poses) else None
            Tp = reader.get_gt_pose(i, par)
            if Op is None or Tp is None or not np.all(np.isfinite(Tp)):
                continue
            a = Op[:3, :3] @ p.joint.axis
            b = Tp[:3, :3] @ ref.axis
            angs.append(float(np.degrees(np.arccos(np.clip(
                abs(float(a @ b)) / max(np.linalg.norm(a) * np.linalg.norm(b),
                                        1e-9), -1, 1)))))
            if p.joint.kind == "revolute" and ref.kind == "revolute" \
                    and p.joint.point is not None and ref.point is not None:
                pa = Op[:3, :3] @ p.joint.point + Op[:3, 3]
                pb = Tp[:3, :3] @ ref.point + Tp[:3, 3]
                w = np.cross(a, b)
                nw = np.linalg.norm(w)
                dv = pb - pa
                dists.append(float(abs(np.dot(dv, w / nw))) if nw > 1e-6 else
                             float(np.linalg.norm(dv - np.dot(dv, b) * b
                                                  / max(b @ b, 1e-9))))
        if not angs:
            continue
        ang = float(np.median(angs))
        d = float(np.median(dists)) if dists else None
        out.append({"part": j, "parent_part": p.parent, "gt_parent": par,
                    "gt": chi, "kind": p.joint.kind, "ref_kind": ref.kind,
                    "spec_kind": gt_type.get(chi), "ang": ang, "dist": d})
    return out


def joint_summary(joints):
    """Axis error and type accuracy over a list of joint_metrics rows."""
    if not joints:
        return None
    ang = np.array([j["ang"] for j in joints])
    typ = [j for j in joints if j["spec_kind"]]
    acc = float(np.mean([j["kind"] == j["spec_kind"] for j in typ])) if typ else float("nan")
    d = [j["dist"] for j in joints if j["dist"] is not None]
    counts = {}
    for row in joints:
        counts[row['gt']] = counts.get(row['gt'], 0) + 1
    return {"n": len(joints), "unique_gt_joints": len(counts),
            "duplicate_gt_rows": sum(n-1 for n in counts.values()),
            "ang_med": float(np.median(ang)),
            "ang_mean": float(ang.mean()), "type_acc": acc,
            "dist_med": float(np.median(d)) if d else None}


def fmt(name, res, n_parts):
    L = [f"{name}: {n_parts} parts vs {res['n_gt']} GT, "
         f"covered {len(res['covered'])}/{res['n_gt']}, "
         f"purity {100 * res['purity']:.1f}%"]
    for g, (tm, rm, tp, rp) in sorted(res["err"].items()):
        L.append(f"    {g:8s} median {tm:6.1f} mm {rm:5.2f} deg   "
                 f"p90 {tp:6.1f} mm {rp:5.2f} deg")
    for j in res.get("joints", []):
        dd = "n/a" if j["dist"] is None else f"{100 * j['dist']:.1f} cm"
        L.append(f"    joint {j['gt']:6s} {j['kind']:9s} (spec {j['spec_kind']}, "
                 f"mocap {j['ref_kind']})  axis {j['ang']:5.1f} deg  dist {dd}")
    return "\n".join(L)


def save_pose_trace(reader, gt, filename):
    """Archive all IDs, including deleted ones, without pickle or tracker feedback."""
    from pathlib import Path
    ids = sorted({p for row in gt['id_log'] for p in row})
    frames = np.array([f for f, _ in gt['log']], dtype=int)
    poses = np.full((len(frames), len(ids), 4, 4), np.nan)
    votes = np.zeros((len(frames), len(ids), len(gt['parts'])), dtype=int)
    observed = np.zeros((len(frames), len(ids)), dtype=bool)
    present = np.zeros_like(observed)
    recovered = np.zeros_like(observed)
    for t, ((_, Ts), row) in enumerate(zip(gt['log'], gt['id_log'])):
        for j, identity in enumerate(row):
            k = ids.index(identity)
            present[t, k] = True
            if Ts[j] is not None:
                poses[t, k] = Ts[j]
            votes[t, k] = gt['votes'][t][j]
            observed[t, k] = gt['observed'][t][j]
            if 'recovered' in gt:
                recovered[t, k] = gt['recovered'][t][j]
    truth = np.full((len(frames), len(gt['parts']), 4, 4), np.nan)
    for t, f in enumerate(frames):
        for k, part in enumerate(gt['parts']):
            T = reader.get_gt_pose(int(f), part)
            if T is not None:
                truth[t, k] = T
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filename, frames=frames, part_ids=ids, poses=poses,
                        present=present, observed=observed, recovered=recovered, votes=votes,
                        gt_names=gt['parts'], gt_poses=truth)

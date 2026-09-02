"""Online part discovery demo: one bbox while rigid, per-part bboxes once split.

Uses TAPIR point tracks (Doersch et al.) and the sequential RANSAC of
Point2Pose (Lin et al., ECCV 2026) directly, bypassing the pipeline.

Feeds an RGB-D stream through the existing Point2Pose front end unchanged --
TAPIR point tracks plus the sequential RANSAC already in
`svd_cluster_ransac_register` -- and adds only the piece the articulated plan
calls for: treating a *persistent secondary consensus set* as a new part.

The demo starts from a single initial mask, samples query points inside it, and
draws one large box while the object still explains as a single rigid body. When
a secondary consensus set survives long enough, the part is spawned and from
then on each part carries its own box.

Nothing here assumes an object category, a joint count, or a CAD model.  Ground
truth is used only to score the result, never to drive it.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from point2pose.data_types.frame import Frame
from point2pose.io.sources.dataset.rbo_reader import RBOReader
from point2pose.modules.register.svd_cluster_ransac_register import (
    SVDClusterRANSACRegister,
)
from point2pose.modules.tracker.tapir_tracker import TapirTracker
from point2pose.utils.visualization import draw_oriented_3d_box

# BGR, chosen to stay distinct against the yellow-lit RBO lab
PART_COLORS = [
    (66, 135, 245),   # base        - amber
    (214, 130, 60),   # part 1      - blue
    (90, 190, 100),   # part 2      - green
    (170, 100, 220),  # part 3      - violet
    (80, 200, 230),   # part 4      - yellow
]
UNASSIGNED = (150, 150, 150)


def sample_query_points(mask, depth, n_points, border=6, min_depth=0.1,
                        mad_k=3.0, min_tol=0.05):
    """Grid-sample points inside the initial mask that also carry valid depth.

    The mask can cover pixels where something else is in front of the object --
    a hand, or the pole in the RBO scenes -- and those pixels carry the
    occluder's depth, not the object's.  Lifting them puts anchor points tens of
    centimetres off, which drags both the registration and the fitted box.  So
    keep only pixels near the dominant depth mode inside the mask, using a
    median/MAD gate rather than a mean so a large occluder cannot shift it.
    """
    m = cv2.erode(mask.astype(np.uint8), np.ones((border, border), np.uint8))
    ys, xs = np.where((m > 0) & (depth > min_depth))
    if len(xs) == 0:
        raise RuntimeError("initial mask has no pixels with valid depth")

    z = depth[ys, xs]
    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    tol = max(mad_k * 1.4826 * mad, min_tol)
    keep = np.abs(z - med) <= tol
    if keep.sum() >= max(32, n_points // 4):
        ys, xs = ys[keep], xs[keep]
    else:
        logging.warning("depth-mode gate kept too few pixels; using all of them")
    if len(xs) <= n_points:
        sel = np.arange(len(xs))
    else:
        sel = np.random.default_rng(0).choice(len(xs), n_points, replace=False)
    return np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # (x, y)


def lift(points_xy, depth, K, min_depth=0.1):
    """Back-project (x, y) image points with the depth at that pixel."""
    H, W = depth.shape
    x = np.clip(np.round(points_xy[:, 0]).astype(int), 0, W - 1)
    y = np.clip(np.round(points_xy[:, 1]).astype(int), 0, H - 1)
    z = depth[y, x]
    valid = z > min_depth
    xyz = np.zeros((len(points_xy), 3), np.float64)
    xyz[:, 0] = (points_xy[:, 0] - K[0, 2]) * z / K[0, 0]
    xyz[:, 1] = (points_xy[:, 1] - K[1, 2]) * z / K[1, 1]
    xyz[:, 2] = z
    return xyz, valid


def fit_oriented_box(points, lo=1.0, hi=99.0, pad=0.005):
    """Oriented 3D box around a part's anchor-frame points.

    Uses a convex-hull minimal box rather than raw PCA. PCA orientation is
    degenerate whenever two extents are similar -- a cabinet front is about
    0.50 x 0.43 m, so any in-plane rotation fits nearly equally well and the box
    comes out visibly skewed. A hull-based box picks the actual edges, and it is
    what the pipeline itself uses for object bboxes.

    Fitted ONCE per part, then carried through the part's estimated pose every
    frame, so the box stays the size of the part instead of the size of whatever
    is currently visible.
    """
    points = np.asarray(points, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 8:
        return None
    # trim outliers before hulling: one bad depth lift moves a hull vertex
    med = np.median(points, axis=0)
    d = np.linalg.norm(points - med, axis=1)
    keep = points[d <= np.percentile(d, hi)]
    if len(keep) < 8:
        keep = points

    try:
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(keep, dtype=np.float64))
        try:
            obb = pcd.get_minimal_oriented_bounding_box()
        except Exception:
            obb = pcd.get_oriented_bounding_box()
        corners = np.asarray(obb.get_box_points())
        # o3d returns hull corners in its own order; reorder to the
        # bottom-face-then-top-face winding draw_oriented_3d_box expects
        c = corners.mean(axis=0)
        R = np.asarray(obb.R)
        local = (corners - c) @ R
        order = np.lexsort((local[:, 0], local[:, 1], local[:, 2]))
        q = local[order]
        bot = q[:4][np.argsort(np.arctan2(q[:4, 1], q[:4, 0]))]
        top = q[4:][np.argsort(np.arctan2(q[4:, 1], q[4:, 0]))]
        local = np.vstack([bot, top])
        local = local * (1.0 + 0.0) + np.sign(local) * pad
        return local @ R.T + c
    except Exception:
        pass

    # fallback: PCA (degenerate orientation, but better than nothing)
    c = keep.mean(axis=0)
    q = keep - c
    if not np.all(np.isfinite(q)) or np.ptp(q) < 1e-9:
        return None
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    R = vt.T
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    proj = q @ R
    lo_e = np.percentile(proj, lo, axis=0) - pad
    hi_e = np.percentile(proj, hi, axis=0) + pad
    corners_local = np.array([
        [lo_e[0], lo_e[1], lo_e[2]], [hi_e[0], lo_e[1], lo_e[2]],
        [hi_e[0], hi_e[1], lo_e[2]], [lo_e[0], hi_e[1], lo_e[2]],
        [lo_e[0], lo_e[1], hi_e[2]], [hi_e[0], lo_e[1], hi_e[2]],
        [hi_e[0], hi_e[1], hi_e[2]], [lo_e[0], hi_e[1], hi_e[2]],
    ])
    return (corners_local @ R.T) + c


def sequential_ransac(register, src, tgt):
    """Run the register's own sequential RANSAC and return every consensus set.

    This is the untouched Point2Pose cluster extraction: the same loop
    `register()` runs before it collapses the candidates down to one pose.
    """
    remaining = np.ones(len(src), dtype=bool)
    clusters = []
    for _ in range(register._max_clusters):
        c = register._RANSAC(
            p0=src, tgt_pcd=tgt, w=None, remaining=remaining, init_pose=None
        )
        if c is None:
            break
        clusters.append(c)
    return clusters


class PartManager:
    """Spawns a part when a secondary consensus set persists across frames.

    Deliberately the simplest rule that matches the plan's spawn condition (b):
    a candidate is followed frame to frame by membership overlap, and is promoted
    once it has survived `persist` consecutive frames with enough support. The
    plan's condition (a) -- a coherent dense residual blob -- is not implemented
    here; this demo exists to test whether (b) alone is a usable signal.
    """

    def __init__(self, n_points, persist=8, min_points=12, overlap=0.4,
                 min_rel_trans=0.015, min_rel_rot_deg=5.0):
        self.persist = persist
        self.min_points = min_points
        self.overlap = overlap
        # A candidate must also be MOVING relative to the base.  Without this the
        # spawn fires on aliasing: consensus sets split off a still-closed drawer
        # purely from depth noise, which is why the plan requires relative motion
        # as well as persistence.
        self.min_rel_trans = min_rel_trans
        self.min_rel_rot_deg = min_rel_rot_deg
        self.assign = np.zeros(n_points, dtype=np.int32)  # everything starts on base
        self.n_parts = 1
        self._candidates = {}   # cid -> {members, age}
        self._next_cid = 0
        self.spawn_log = []     # (frame, part_id, n_points)
        self.box_local = {}     # pid -> (8,3) corners in the anchor frame
        self.pose = {}          # pid -> last known 4x4 anchor->camera

    @staticmethod
    def _jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    @staticmethod
    def _rel_motion(T_base, T_cand):
        """Translation (m) and rotation (deg) of a candidate relative to the base."""
        rel = np.linalg.inv(T_base) @ T_cand
        trans = float(np.linalg.norm(rel[:3, 3]))
        cos = (np.trace(rel[:3, :3]) - 1.0) / 2.0
        rot = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return trans, rot

    def track_poses(self, clusters, min_overlap=0.15):
        """Attach each part to the cluster that best explains it.

        A part with no matching cluster this frame (fully occluded, say) keeps its
        last known pose rather than vanishing, which is what makes the projected
        box survive occlusion.
        """
        for pid in range(self.n_parts):
            members = set(np.where(self.assign == pid)[0].tolist())
            if not members:
                continue
            best_T, best_j = None, min_overlap
            for c in clusters:
                inl = set(c["inliers"].tolist())
                j = len(members & inl) / max(1, len(members | inl))
                if j > best_j:
                    best_T, best_j = c["T"], j
            if best_T is not None:
                self.pose[pid] = best_T

    def refit_boxes(self, anchor_xyz, anchor_valid):
        for pid in range(self.n_parts):
            idx = np.where((self.assign == pid) & anchor_valid)[0]
            box = fit_oriented_box(anchor_xyz[idx])
            if box is not None:
                self.box_local[pid] = box

    def update(self, frame_idx, clusters):
        """clusters are ordered largest-first; cluster 0 is taken as the base."""
        if len(clusters) < 2:
            self._candidates = {k: v for k, v in self._candidates.items()}
            return

        T_base = clusters[0]["T"]
        secondary = [set(c["inliers"].tolist()) for c in clusters[1:]]
        moving = []
        for c in clusters[1:]:
            t, r = self._rel_motion(T_base, c["T"])
            moving.append(t >= self.min_rel_trans or r >= self.min_rel_rot_deg)
        matched, fresh = set(), {}

        for members, is_moving in zip(secondary, moving):
            best_cid, best_j = None, 0.0
            for cid, cand in self._candidates.items():
                if cid in matched:
                    continue
                j = self._jaccard(members, cand["members"])
                if j > best_j:
                    best_cid, best_j = cid, j

            if best_cid is not None and best_j >= self.overlap:
                matched.add(best_cid)
                cand = self._candidates[best_cid]
                fresh[best_cid] = {
                    "members": cand["members"] | members,
                    # age only advances on frames where the candidate is actually
                    # moving relative to the base
                    "age": cand["age"] + (1 if is_moving else 0),
                }
            else:
                fresh[self._next_cid] = {"members": members,
                                         "age": 1 if is_moving else 0}
                self._next_cid += 1

        # promote whatever has now survived long enough
        for cid, cand in list(fresh.items()):
            if cand["age"] >= self.persist and len(cand["members"]) >= self.min_points:
                idx = np.array(sorted(cand["members"]), dtype=int)
                # merge into an existing non-base part if they largely coincide,
                # rather than fragmenting one drawer across many part ids
                merged_into = None
                for pid_existing in range(1, self.n_parts):
                    have = set(np.where(self.assign == pid_existing)[0].tolist())
                    if self._jaccard(cand["members"], have) >= 0.3:
                        merged_into = pid_existing
                        break
                if merged_into is not None:
                    self.assign[idx] = merged_into
                    fresh.pop(cid)
                    continue
                pid = self.n_parts
                self.n_parts += 1
                self.assign[idx] = pid
                self.spawn_log.append(
                    {"frame": frame_idx, "part_id": pid, "n_points": int(len(idx))}
                )
                fresh.pop(cid)
        self._candidates = fresh

    @property
    def discovered(self):
        return self.n_parts > 1


def _project(corners, K, T):
    pts = (T @ np.concatenate([corners, np.ones((len(corners), 1))], 1).T).T[:, :3]
    if np.any(pts[:, 2] < 1e-6):
        return None
    uv = (K @ pts.T).T
    return uv[:, :2] / uv[:, 2:3]


def draw(rgb, tracks, visible, assign, n_parts, frame_idx, status, pm, K,
         spawn_flash=0, box_mode="3d"):
    vis = rgb.copy()
    H, W = vis.shape[:2]

    for pid in range(n_parts):
        color = PART_COLORS[pid % len(PART_COLORS)]
        thick = 3 if (pid > 0 and spawn_flash > 0) else 2
        tag = "object (rigid)" if (pid == 0 and n_parts == 1) else (
            "base" if pid == 0 else f"part {pid}")

        anchor = None
        box, T = pm.box_local.get(pid), pm.pose.get(pid)
        if box_mode in ("3d", "both") and box is not None and T is not None:
            uv = _project(box, K, T)
            if uv is not None:
                vis = draw_oriented_3d_box(K, vis, T, box, line_color=color,
                                           linewidth=thick)
                anchor = (int(uv[:, 0].min()), int(uv[:, 1].min()))

        if box_mode in ("2d", "both") or anchor is None:
            sel = (assign == pid) & visible
            if sel.sum() < 4:
                continue
            pts = tracks[sel]
            x0, y0 = np.percentile(pts, 2, axis=0)
            x1, y1 = np.percentile(pts, 98, axis=0)
            x0, y0 = int(max(0, x0 - 10)), int(max(0, y0 - 10))
            x1, y1 = int(min(W - 1, x1 + 10)), int(min(H - 1, y1 + 10))
            if box_mode in ("2d", "both"):
                cv2.rectangle(vis, (x0, y0), (x1, y1), color, 1)
            if anchor is None:
                anchor = (x0, y0)

        x0, y0 = int(np.clip(anchor[0], 0, W - 90)), int(np.clip(anchor[1], 16, H - 4))
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x0, y0 - th - 8), (x0 + tw + 8, y0), color, -1)
        cv2.putText(vis, tag, (x0 + 4, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (20, 20, 20), 1, cv2.LINE_AA)

    # Filled dot = TAPIR reports the point visible; hollow ring = it reports the
    # point occluded.  Occluded points are still drawn so the occlusion itself is
    # legible, but they take no part in registration.
    for i in range(len(tracks)):
        c = PART_COLORS[assign[i] % len(PART_COLORS)] if assign[i] >= 0 else UNASSIGNED
        pt = tuple(np.round(tracks[i]).astype(int))
        if not (0 <= pt[0] < W and 0 <= pt[1] < H):
            continue
        if visible[i]:
            cv2.circle(vis, pt, 3, c, -1, cv2.LINE_AA)
        else:
            cv2.circle(vis, pt, 3, c, 1, cv2.LINE_AA)

    bar = np.zeros((34, W, 3), np.uint8)
    bar[:] = (28, 24, 20)
    cv2.putText(bar, f"frame {frame_idx:04d}", (10, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, status, (140, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (120, 220, 140) if "part" in status else (200, 200, 200), 1, cv2.LINE_AA)

    lx = W - 250
    cv2.circle(bar, (lx, 17), 3, (235, 235, 235), -1, cv2.LINE_AA)
    cv2.putText(bar, "visible", (lx + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.circle(bar, (lx + 90, 17), 3, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(bar, "occluded", (lx + 100, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1, cv2.LINE_AA)
    n_occ = int((~visible).sum())
    cv2.putText(bar, f"{n_occ} occluded", (W - 380, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (170, 170, 170), 1, cv2.LINE_AA)
    return np.vstack([vis, bar])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--models-dir", default=None)
    ap.add_argument("--out", default="debug/multi-parts/part_discovery.mp4")
    ap.add_argument("--n-points", type=int, default=400)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-rel-trans", type=float, default=0.015,
                    help="relative translation vs base required to call a part moving")
    ap.add_argument("--persist", type=int, default=8,
                    help="frames a secondary consensus set must survive to spawn a part")
    ap.add_argument("--inlier-thres", type=float, default=0.004)
    ap.add_argument("--min-inliers", type=int, default=5)
    ap.add_argument("--ransac-iters", type=int, default=100)
    ap.add_argument("--max-clusters", type=int, default=4)
    ap.add_argument("--checkpoint", default="checkpoints/tapir/causal_bootstapir_checkpoint.pt")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--box-mode", choices=["3d", "2d", "both"], default="3d",
                    help="3d projects an oriented box through the part pose; "
                         "2d is the old percentile box over visible tracks")
    ap.add_argument("--save-tracks", default=None,
                    help="npz of tracks/depth-lifted points, for offline parameter sweeps")
    args = ap.parse_args()

    reader = RBOReader(args.seq_dir, models_dir=args.models_dir)
    gt_parts = reader.get_object_names()
    print(f"[demo] {reader.get_video_name()}  {len(reader)} frames  GT parts {gt_parts}")

    frames_idx = list(range(0, len(reader), args.stride))
    if args.max_frames:
        frames_idx = frames_idx[: args.max_frames]

    # ---- frame 0: initial mask -> query points -------------------------------
    rgb0, depth0 = reader.get_color(frames_idx[0]), reader.get_depth(frames_idx[0])
    init_mask = np.zeros(rgb0.shape[:2], np.uint8)
    for m in reader.get_masks(frames_idx[0]):
        init_mask |= m
    pts0 = sample_query_points(init_mask, depth0, args.n_points)
    print(f"[demo] initial mask covers {100*init_mask.mean():.1f}% of the image; "
          f"sampled {len(pts0)} query points")

    # GT part label per query point, used ONLY for scoring
    pidx = reader.render_part_index_map(frames_idx[0])
    gt_label = pidx[np.round(pts0[:, 1]).astype(int), np.round(pts0[:, 0]).astype(int)]

    tracker = TapirTracker({
        "checkpoint_path": args.checkpoint,
        "resize_height": 480, "resize_width": 480,
        "visible_threshold": 0.5, "device": "cuda",
    })
    f0 = Frame(id=0, rgb=rgb0, depth=depth0, intrinsics=reader.K)
    tracker.add_query_points(f0, pts0)
    tracker.initialize(f0)

    register = SVDClusterRANSACRegister({
        "ransac_iters": args.ransac_iters,
        "sample_size": 4,
        "inlier_thres": args.inlier_thres,
        "min_inliers": args.min_inliers,
        "max_clusters": args.max_clusters,
        "use_uncertainty": False,
    })

    anchor_xyz, anchor_valid = lift(pts0, depth0, reader.K)
    pm = PartManager(len(pts0), persist=args.persist,
                     min_rel_trans=args.min_rel_trans)
    pm.refit_boxes(anchor_xyz, anchor_valid)
    pm.pose[0] = np.eye(4)
    prev_sets = []

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer, flash = None, 0
    history = []
    rec = {"xyz": [], "valid": [], "visible": [], "tracks": []}

    for n, i in enumerate(frames_idx):
        rgb, depth = reader.get_color(i), reader.get_depth(i)
        frame = Frame(id=i, rgb=rgb, depth=depth, intrinsics=reader.K)
        tracks, _unc, visible = tracker.track_once(frame)
        visible = visible.astype(bool)

        cur_xyz, cur_valid = lift(tracks, depth, reader.K)
        usable = np.where(anchor_valid & cur_valid & visible)[0]
        if args.save_tracks:
            rec["xyz"].append(cur_xyz); rec["valid"].append(cur_valid)
            rec["visible"].append(visible); rec["tracks"].append(tracks)

        clusters = []
        if len(usable) >= max(args.min_inliers, 8):
            # Registration is anchor -> current (a long baseline, as frame-to-map
            # does).  Consecutive frames move a joint by far less than the inlier
            # threshold, so parts are not separable frame to frame.
            local = sequential_ransac(register, anchor_xyz[usable], cur_xyz[usable])
            for c in local:  # map local indices back to global point ids
                c["inliers"] = usable[c["inliers"]]
                clusters.append(c)
            clusters.sort(key=lambda c: -c["ninliers"])

        # --- diagnostics: is a secondary consensus set actually persistent? ---
        comp = []
        for c in clusters:
            lab = gt_label[c["inliers"]]
            cnt = defaultdict(int)
            for l in lab:
                cnt[gt_parts[l] if 0 <= l < len(gt_parts) else "bg"] += 1
            dom, dn = max(cnt.items(), key=lambda kv: kv[1])
            comp.append({"n": int(c["ninliers"]), "dominant_gt": dom,
                         "purity": round(dn / len(lab), 3)})
        cur_sets = [set(c["inliers"].tolist()) for c in clusters]
        jac = []
        for si, s_ in enumerate(cur_sets):
            best = max((len(s_ & p_) / len(s_ | p_) for p_ in prev_sets), default=0.0)
            jac.append(round(best, 3))
        prev_sets = cur_sets

        before = pm.n_parts
        pm.update(i, clusters)
        if pm.n_parts != before or not pm.box_local:
            pm.refit_boxes(anchor_xyz, anchor_valid)
        pm.track_poses(clusters)
        if pm.n_parts > before:
            flash = 10
            print(f"[demo] frame {i}: PART SPAWNED -> {pm.n_parts} parts "
                  f"({pm.spawn_log[-1]['n_points']} points)")

        status = (f"{pm.n_parts} parts tracked" if pm.discovered
                  else "single rigid body")
        canvas = draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), tracks, visible,
                      pm.assign, pm.n_parts, i, status, pm, reader.K, flash,
                      box_mode=args.box_mode)
        flash = max(0, flash - 1)

        if writer is None:
            h, w = canvas.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (w, h))
        writer.write(canvas)

        history.append({
            "frame": int(i),
            "n_usable": int(len(usable)),
            "n_clusters": len(clusters),
            "cluster_sizes": [int(c["ninliers"]) for c in clusters],
            "cluster_gt": comp,
            "cluster_jaccard_prev": jac,
            "n_parts": int(pm.n_parts),
            "joint_states": reader.get_joint_states(i),
        })
        if n % 25 == 0:
            desc = "  ".join(f"{c['n']}@{c['dominant_gt']}({c['purity']:.2f})/J{j}"
                             for c, j in zip(comp, jac))
            print(f"  frame {i:4d}  usable {len(usable):3d}  parts {pm.n_parts}  {desc}")

    if writer:
        writer.release()

    if args.save_tracks:
        np.savez_compressed(
            args.save_tracks,
            anchor_xyz=anchor_xyz, anchor_valid=anchor_valid, gt_label=gt_label,
            frames=np.array(frames_idx[: len(rec["xyz"])]),
            gt_parts=np.array(gt_parts),
            **{k: np.stack(v) for k, v in rec.items()},
        )
        print(f"[demo] wrote {args.save_tracks}")

    # ---- score the discovery against ground truth ----------------------------
    report = {"sequence": reader.get_video_name(), "gt_parts": gt_parts,
              "spawns": pm.spawn_log, "history": history}
    print(f"\n[demo] wrote {args.out}")
    print(f"[demo] discovered {pm.n_parts} parts (GT has {len(gt_parts)})")
    for pid in range(pm.n_parts):
        sel = pm.assign == pid
        if sel.sum() == 0:
            continue
        labels = gt_label[sel]
        counts = defaultdict(int)
        for l in labels:
            counts[gt_parts[l] if 0 <= l < len(gt_parts) else "background"] += 1
        dom, dom_n = max(counts.items(), key=lambda kv: kv[1])
        purity = dom_n / len(labels)
        print(f"  part {pid}: {sel.sum():3d} pts   dominant GT = {dom:12s} "
              f"purity {purity*100:5.1f}%   {dict(counts)}")
        report.setdefault("parts", []).append({
            "part_id": pid, "n_points": int(sel.sum()),
            "dominant_gt": dom, "purity": float(purity),
            "composition": {k: int(v) for k, v in counts.items()},
        })

    js = reader.get_joint_state_trajectory()
    for s in pm.spawn_log:
        k = frames_idx.index(s["frame"]) if s["frame"] in frames_idx else 0
        moved = {n: float(abs(v[: k + 1] - v[0]).max()) for n, v in js.items()}
        print(f"  spawn at frame {s['frame']}: joint travel so far "
              + ", ".join(f"{n}={m*1000:.0f}mm" for n, m in moved.items()))
        s["joint_travel_at_spawn_mm"] = {n: m * 1000 for n, m in moved.items()}

    out_json = os.path.splitext(args.out)[0] + "_report.json"
    json.dump(report, open(out_json, "w"), indent=2)
    print(f"[demo] wrote {out_json}")


if __name__ == "__main__":
    main()

"""Geometry evidence for new views; no appearance or point-ID correspondence."""
import cv2
import numpy as np


def uncovered_surface(mask, depth, tracks, radius=14, min_region=400):
    """Visible mask regions not covered by currently reliable image tracks."""
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.05)
    seeds = np.ones(mask.shape, np.uint8)
    pts = np.asarray(tracks).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)].round().astype(int)
    h, w = mask.shape
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < w)
              & (pts[:, 1] >= 0) & (pts[:, 1] < h)]
    if len(pts):
        seeds[pts[:, 1], pts[:, 0]] = 0
        valid &= cv2.distanceTransform(seeds, cv2.DIST_L2, 5) > radius
    count, labels, stats, _ = cv2.connectedComponentsWithStats(valid.astype(np.uint8))
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_region
    return keep[labels].astype(np.uint8)


def match_joint_surface(points, poses, depth, mask, K, tolerance=0.012,
                        min_support=20, min_fraction=0.3, margin=0.05):
    """Find a distinguishable joint pose from depth, with no feature matches.

    Score the same canonical surface at every candidate pose. Geometry behind
    observed depth may be occluded and is neutral; geometry in observed free
    space is penalized. A two-sided thin panel needs no normal or RGB matching.
    Different poses with indistinguishable support are explicitly rejected.
    """
    points, poses = np.asarray(points), np.asarray(poses)
    if len(points) < min_support or not len(poses):
        return None
    good = np.isfinite(points).all(axis=1)
    points = points[good]
    if len(points) < min_support:
        return None
    q = np.einsum("mij,nj->mni", poses[:, :3, :3], points) + poses[:, None, :3, 3]
    z = q[..., 2]
    safe = np.maximum(z, 1e-6)
    u = np.rint(q[..., 0] * K[0, 0] / safe + K[0, 2]).astype(int)
    v = np.rint(q[..., 1] * K[1, 1] / safe + K[1, 2]).astype(int)
    h, w = depth.shape
    valid = (z > 0.05) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uc, vc = np.clip(u, 0, w - 1), np.clip(v, 0, h - 1)
    observed = depth[vc, uc]
    valid &= np.isfinite(observed) & (observed > 0.05)
    support = valid & (mask[vc, uc] > 0) & (np.abs(z - observed) < tolerance)
    free = valid & (z < observed - tolerance)
    counts = support.sum(axis=1)
    scores = (counts - 0.5 * free.sum(axis=1)) / len(points)
    best = int(scores.argmax())
    if counts[best] < min_support or scores[best] < min_fraction:
        return None
    # Adjacent grid samples describe the same noisy surface; compare against
    # candidates whose predicted geometry is meaningfully different instead.
    apart = np.median(np.linalg.norm(q - q[best], axis=-1), axis=1) > 2 * tolerance
    runner = float(scores[apart].max()) if apart.any() else -np.inf
    if scores[best] - runner < margin:
        return None
    region = np.zeros_like(mask, np.uint8)
    keep = support[best]
    region[vc[best, keep], uc[best, keep]] = 1
    region = cv2.dilate(region, np.ones((9, 9), np.uint8))
    region &= ((mask > 0) & (depth > 0.05)).astype(np.uint8)
    return {"index": best, "pose": poses[best].copy(),
            "score": float(scores[best]), "support": int(counts[best]),
            "region": region}


def reprojection_evidence(points, pose, depth, mask, K, tolerance=.012,
                          occlusion_aware=True):
    """Depth support and free-space contradiction for a fixed canonical sample.

    Behind-surface geometry is unobserved, not evidence of a second motion.
    Invalid depth and out-of-frame samples are neutral. Valid background pixels
    contradict the predicted object silhouette. Fractions use the SAME sample
    denominator across competing pose hypotheses.
    """
    points = np.asarray(points)
    if not len(points):
        return {"support": 0., "contradiction": 0., "visible": 0.}
    q = points @ pose[:3, :3].T + pose[:3, 3]
    z = q[:, 2]
    uv = q @ np.asarray(K).T
    xy = np.rint(uv[:, :2] / np.maximum(z[:, None], 1e-6)).astype(int)
    h, w = depth.shape
    valid = np.isfinite(q).all(axis=1) & (z > .05) & \
        (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    u, v = np.clip(xy[:, 0], 0, w - 1), np.clip(xy[:, 1], 0, h - 1)
    d = depth[v, u]
    valid &= np.isfinite(d) & (d > .05)
    obj = mask[v, u] > 0
    support = valid & obj & (np.abs(z - d) <= tolerance)
    contradiction = valid & ((z < d - tolerance) | (~obj & (z <= d + tolerance)))
    if not occlusion_aware:
        contradiction |= valid & (z > d + tolerance)
    return {"support": float(support.mean()),
            "contradiction": float(contradiction.mean()),
            "visible": float((support | contradiction).mean())}

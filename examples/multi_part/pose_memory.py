"""Causal pose hypotheses scored against persistent geometry, never GT poses."""
import numpy as np
from examples.multi_part.surface_memory import reprojection_evidence


def joint_pose_supported(points, free_pose, joint_pose, depth, mask, K,
                         min_support=.25, slack=.02):
    """A joint constraint may not override better observed surface evidence."""
    if len(points) < 40 or not np.isfinite(joint_pose).all():
        return False
    free = reprojection_evidence(points, free_pose, depth, mask, K)
    joint = reprojection_evidence(points, joint_pose, depth, mask, K)
    return bool(joint['support'] >= min_support and
                joint['support'] - joint['contradiction'] + slack >=
                free['support'] - free['contradiction'])


def choose_pose(points, sparse_pose, predicted_pose, depth, mask, K,
                refine, tolerance=.012, margin=.08, min_support=.3,
                previous_pose=None, max_step_deg=20.):
    """Rescue a contradicted sparse fit only when refined temporal geometry wins.

    `refine` is a local geometric optimizer supplied by the caller. Both raw and
    refined predictions must compete on the same stored surface sample. This
    does not create pose evidence if the surface is entirely hidden.
    """
    def score(T):
        evidence = reprojection_evidence(points, T, depth, mask, K, tolerance)
        return evidence['support'] - evidence['contradiction'], evidence['support']
    def continuous(T):
        if previous_pose is None:
            return True
        R = T[:3, :3] @ previous_pose[:3, :3].T
        angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        return angle <= max_step_deg
    sparse_continuous = continuous(sparse_pose)
    base, _ = score(sparse_pose)
    # Avoid invoking dense optimization for an already well-supported fit.
    if base >= .85 and sparse_continuous:
        return sparse_pose, False
    refined = refine(predicted_pose)
    candidates = [predicted_pose, refined]
    best = sparse_pose
    best_score = base
    rescued = False
    for T in candidates:
        if not np.isfinite(T).all():
            continue
        value, support = score(T)
        continuity_rescue = not sparse_continuous and continuous(T)
        if support >= min_support and continuous(T) and (
                (continuity_rescue and not rescued) or
                (value > best_score and (continuity_rescue or value > base + margin))):
            best, best_score, rescued = T, value, True
    return best, rescued


def incremental_candidate(previous, current, pose, register, max_step_deg=20.,
                          max_residual=.012):
    """Compose adjacent-frame motion without consuming the split sampler's RNG."""
    if len(previous) < 3:
        return None
    rng = np.random.get_state()
    try:
        result = register._RANSAC(p0=previous, tgt_pcd=current, w=None,
                                 remaining=np.ones(len(previous), bool), init_pose=None)
    finally:
        np.random.set_state(rng)
    if result is None:
        return None
    delta = np.asarray(result['T'])
    if not np.isfinite(delta).all():
        return None
    angle = np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1) / 2, -1, 1)))
    residual = np.median(np.linalg.norm(previous @ delta[:3, :3].T + delta[:3, 3] - current, axis=1))
    if angle >= max_step_deg or residual >= max_residual:
        return None
    return delta @ pose

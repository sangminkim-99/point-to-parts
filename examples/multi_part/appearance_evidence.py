"""CPU appearance evidence for RGB-D pose hypotheses; no map mutation.

A diagnostic baseline on visible Gaussian centres, not a rendered RGB loss.
"""
import numpy as np


def appearance_evidence(points, colors, pose, K, rgb, depth, mask, tolerance=.012, return_samples=False):
    points, colors = np.asarray(points), np.asarray(colors)
    if points.shape != colors.shape or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points/colors must have matching (N,3) shapes')
    h,w = depth.shape
    if rgb.shape != (h,w,3) or mask.shape != (h,w):
        raise ValueError('RGB, depth and mask must be aligned')
    q = points @ np.asarray(pose)[:3,:3].T + np.asarray(pose)[:3,3]
    valid = np.isfinite(q).all(1) & np.isfinite(colors).all(1) & (q[:,2] > .05)
    ids = np.flatnonzero(valid)
    uv = q[ids] @ np.asarray(K).T
    xy = np.rint(uv[:,:2]/uv[:,2:]).astype(int)
    inside = (xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    ids,xy = ids[inside],xy[inside]
    # One nearest surface per pixel; do not count hidden duplicates as votes.
    order = np.argsort(q[ids,2],kind='stable')
    ids,xy = ids[order],xy[order]
    _, first = np.unique(xy[:,1]*w+xy[:,0],return_index=True)
    ids,xy = ids[first],xy[first]
    observed_depth = depth[xy[:,1],xy[:,0]]
    supported = (mask[xy[:,1],xy[:,0]]>0)&np.isfinite(observed_depth)&(observed_depth>.05)&(np.abs(q[ids,2]-observed_depth)<=tolerance)
    ids,xy = ids[supported],xy[supported]
    if not len(ids):
        return dict(samples=0, mean_rgb_l1=None, median_rgb_l1=None,
                    **({'ids': ids, 'errors': np.empty(0)} if return_samples else {}))
    observed = np.asarray(rgb,dtype=float)[xy[:,1],xy[:,0]] / 255.
    error = np.abs(np.clip(colors[ids],0,1)-observed).mean(1)
    return dict(samples=len(ids), mean_rgb_l1=float(error.mean()), median_rgb_l1=float(np.median(error)),
                **({'ids': ids, 'errors': error} if return_samples else {}))


def paired_appearance_evidence(points, colors, first_pose, second_pose, K, rgb,
                               depth, mask, tolerance=.012, min_samples=20):
    """Compare the SAME stored samples visible under both poses.

    Positive gain favors the second candidate. No decision is made: even a
    paired improvement can hide different unsupported regions, so support counts
    accompany the score and must be considered by a consumer.
    """
    first = appearance_evidence(points, colors, first_pose, K, rgb, depth, mask,
                                tolerance, return_samples=True)
    second = appearance_evidence(points, colors, second_pose, K, rgb, depth, mask,
                                 tolerance, return_samples=True)
    ids, a, b = np.intersect1d(first['ids'], second['ids'], return_indices=True)
    enough = len(ids) >= max(1, min_samples)
    return {'first_samples': first['samples'], 'second_samples': second['samples'],
            'paired_samples': len(ids),
            'first_rgb_l1': float(first['errors'][a].mean()) if enough else None,
            'second_rgb_l1': float(second['errors'][b].mean()) if enough else None,
            'gain': float((first['errors'][a]-second['errors'][b]).mean()) if enough else None}

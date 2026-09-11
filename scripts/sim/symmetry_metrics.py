"""Evaluation-only surface displacement under explicitly supplied symmetries.

Inputs are absolute link-to-camera poses and object-link-local surface samples.
Never infer symmetries from predictions, and never feed these into tracking.
"""
import numpy as np


def surface_pose_errors(predicted, truth, points, symmetries=()):
    """Return strict and symmetry-aware RMS surface errors in metres.

    Each symmetry maps the object link frame to itself. Identity is always
    included. Symmetry alternatives use truth @ S, preserving off-origin
    symmetry centers. The complete transform is scored jointly, not independent
    minima for rotation and translation from different alternatives.
    """
    predicted, truth = np.asarray(predicted, float), np.asarray(truth, float)
    points = np.asarray(points, float)
    if predicted.shape != truth.shape or predicted.ndim != 3 or predicted.shape[1:] != (4,4):
        raise ValueError('poses must have matching (frames,4,4) shapes')
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError('need nonempty (points,3) object-local surface samples')
    choices = [np.eye(4)] + [np.asarray(s, float) for s in symmetries]
    for T in (predicted, truth, np.array(choices)):
        if not np.isfinite(T).all() or T.shape[-2:] != (4,4):
            raise ValueError('invalid transforms')
        if not np.allclose(T[...,3,:], [0,0,0,1]):
            raise ValueError('invalid homogeneous row')
        R = T[...,:3,:3]
        if not np.allclose(R.swapaxes(-1,-2) @ R, np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(R),1,atol=1e-5):
            raise ValueError('transforms must be proper rigid motions')
    if not np.isfinite(points).all():
        raise ValueError('invalid surface samples')
    predicted_points = points @ predicted[:,:3,:3].transpose(0,2,1) + predicted[:,None,:3,3]
    errors=[]
    for S in choices:
        equivalent = truth @ S
        target = points @ equivalent[:,:3,:3].transpose(0,2,1) + equivalent[:,None,:3,3]
        errors.append(np.sqrt(np.mean(np.sum((predicted_points-target)**2,axis=-1),axis=-1)))
    errors=np.array(errors)
    return {'strict_rms_m': errors[0], 'symmetry_rms_m': errors.min(axis=0),
            'symmetry_index': errors.argmin(axis=0)}

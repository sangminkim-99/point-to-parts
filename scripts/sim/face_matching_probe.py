"""Opposite-face matching failure: symmetric vs distinguishable faces.

Runs the unmodified tracker on a turnover rendered twice -- identical geometry,
seed and motion, differing ONLY in face appearance (`--faces uniform` vs
`distinct`) -- and scores both with scripts/sim/symmetry_metrics.py
(evaluation-only, root-owned, imported unmodified).

Pose handling, per that module's contract: the tracker's pose is relative to
its own anchor frame, NOT an absolute link-to-camera pose. It is aligned to
the initial GT before scoring:

    predicted_abs(t) = rec(t) @ inv(rec(0)) @ gt(0)

so predicted_abs(0) == gt(0) by construction and every later frame is the
tracker's recovered motion applied to the true initial pose.

Symmetries are DECLARED from the known box geometry -- the proper symmetry
group of a cuboid (180-degree rotations about its three principal axes),
expressed in the link frame at the slab centre -- never inferred from errors.
Strict RMS is always reported alongside; for the distinct-faces slab the
geometric symmetries still hold for the SHAPE, so the same declared set is
used for both, and the report never picks a metric based on which is lower.

    python -m scripts.sim.face_matching_probe <uniform_dir> <distinct_dir> --out <dir>
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from scripts.sim.symmetry_metrics import surface_pose_errors


def box_symmetries():
    """Proper symmetry group of a cuboid (D2): 180-deg turns about x, y, z."""
    out = []
    for axis in range(3):
        S = -np.eye(3)
        S[axis, axis] = 1.0
        T = np.eye(4)
        T[:3, :3] = S
        out.append(T)
    return out


def surface_samples(body_size, n_edge=5):
    """Object-local samples on all six faces of the slab."""
    hx, hy, hz = np.asarray(body_size, float) / 2
    u = np.linspace(-1, 1, n_edge)
    pts = []
    for a, b in [(x, y) for x in u for y in u]:
        pts += [[a * hx, b * hy, hz], [a * hx, b * hy, -hz],
                [a * hx, hy, b * hz], [a * hx, -hy, b * hz],
                [hx, a * hy, b * hz], [-hx, a * hy, b * hz]]
    return np.unique(np.round(np.array(pts), 9), axis=0)


def run_tracker(render_dir, checkpoint, config, overrides):
    from examples.multi_part.naive import NaiveConfig
    from examples.multi_part.acquire import apply_config, apply_overrides
    from examples.multi_part.author_demo import make_tracker
    from examples.multi_part.author_state import union_mask
    render_dir = Path(render_dir)
    meta = json.loads((render_dir / 'meta.json').read_text())
    K = np.array(meta['intrinsics'], float)
    names = sorted(p.name for p in (render_dir / 'rgb').glob('*.png'))
    cfg = apply_config(NaiveConfig(), config)
    apply_overrides(cfg, overrides)
    stream = make_tracker(K, cfg, checkpoint)
    rec, observed, parts = [], [], []
    for i, name in enumerate(names):
        rgb = cv2.cvtColor(cv2.imread(str(render_dir / 'rgb' / name)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(render_dir / 'depth' / name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
        mask = union_mask(cv2.imread(str(render_dir / 'seg' / name), cv2.IMREAD_GRAYSCALE))
        if i == 0:
            stream.start(rgb, depth, mask)
            stream.root = 0
        else:
            stream.step(rgb, depth, mask)
        root = stream.parts[stream.root]
        rec.append(root.pose.copy())
        observed.append(bool(root.observed))
        parts.append(len(stream.parts))
    return np.stack(rec), observed, parts, meta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('uniform_dir')
    ap.add_argument('distinct_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/tapir/causal_bootstapir_checkpoint.pt')
    ap.add_argument('--config', default='reprojection_split.yaml')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    result = dict(symmetries='declared D2 of the cuboid (180 deg about x, y, z), '
                             'from known geometry; NEVER inferred from errors',
                  alignment='predicted_abs(t) = rec(t) @ inv(rec(0)) @ gt(0)',
                  gt_note='GT poses eval-only; tracker input rgb/depth + oracle union mask',
                  runs={})
    for name, rdir in (('uniform', a.uniform_dir), ('distinct', a.distinct_dir)):
        gt = np.load(Path(rdir) / 'poses.npz')['T_cam_part'][:, 0]
        rec, observed, parts, meta = run_tracker(rdir, a.checkpoint, a.config, ['dense=true'])
        pred_abs = rec @ np.linalg.inv(rec[0]) @ gt[0]          # absolute link-to-camera
        pts = surface_samples(meta['body_size_m'])
        err = surface_pose_errors(pred_abs, gt, pts, box_symmetries())
        strict, symm, idx = err['strict_rms_m'], err['symmetry_rms_m'], err['symmetry_index']
        obs = np.array(observed)
        result['runs'][name] = dict(
            render_dir=str(rdir), frames=len(gt), parts_max=max(parts),
            observed_coverage=float(obs.mean()),
            surface_points=len(pts),
            strict_rms_m=dict(median=float(np.median(strict)), max=float(strict.max()),
                              final=float(strict[-1])),
            symmetry_rms_m=dict(median=float(np.median(symm)), max=float(symm.max()),
                                final=float(symm[-1])),
            final_symmetry_index=int(idx[-1]),
            frames_where_symmetry_explains=int(np.sum((strict > 0.02) & (symm < 0.02))),
            per_frame=dict(strict_rms_m=strict.tolist(), symmetry_rms_m=symm.tolist(),
                           symmetry_index=idx.tolist(), observed=observed))
        print(f'{name}: strict RMS median {np.median(strict)*1000:.1f} mm, final '
              f'{strict[-1]*1000:.1f} mm | symmetry-aware final {symm[-1]*1000:.1f} mm '
              f'(index {idx[-1]}) | parts_max {max(parts)}', flush=True)
    (out / 'face_matching.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(f'[faces] wrote {out}/face_matching.json')


if __name__ == '__main__':
    main()

"""Measure Gaussian growth/carve vs GT new-view coverage under rigid motion.

User report: the dense model DUPLICATES geometry during vertical whole-object
motion. Two explanations compete:
  (a) duplication -- the pose does not absorb the motion, so already-modelled
      surface re-enters at a new location and is added again;
  (b) real coverage -- the motion genuinely reveals surface never seen before,
      and growth is correct.

This probe separates them with GT that stays EVAL-ONLY: per frame it counts
5 mm OBJECT-FRAME surface voxels (masked depth back-projected through the GT
pose) that were never visible before -- the ceiling on legitimate growth. The
tracker itself receives only rgb, depth and the binary union mask. On a pure
vertical lift of a rigid slab with a fixed camera the same faces stay visible,
so GT new-view coverage is ~0 by construction and ANY sustained growth is
duplication.

Runs the unmodified tracker twice: baseline, and growth disabled via the
existing config gate (grow_every=0; naive.py:576 skips growth entirely).
Reports per-frame model size, assigned/unassigned, pose error (frame-0-relative
vs GT), and the GT new-voxel profile.

    python -m scripts.sim.gaussian_growth_probe <render_dir> --out <dir>
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np


def gt_new_voxels(render_dir, voxel=0.005, stride=3):
    """Per-frame count of object-frame surface voxels never seen before.

    Uses GT poses + rendered depth + the union mask -- evaluation only.
    """
    render_dir = Path(render_dir)
    meta = json.loads((render_dir / 'meta.json').read_text())
    K = np.array(meta['intrinsics'])
    gt = np.load(render_dir / 'poses.npz')['T_cam_part'][:, 0]
    names = sorted(p.name for p in (render_dir / 'rgb').glob('*.png'))
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    seen, new_per_frame, total_per_frame = set(), [], []
    for i, name in enumerate(names):
        depth = cv2.imread(str(render_dir / 'depth' / name), cv2.IMREAD_UNCHANGED).astype(np.float64) / 1000.
        seg = cv2.imread(str(render_dir / 'seg' / name), cv2.IMREAD_GRAYSCALE)
        m = (seg != 255) & (depth > 0)
        ys, xs = np.where(m)
        ys, xs = ys[::stride], xs[::stride]
        z = depth[ys, xs]
        pc = np.stack([(xs + .5 - cx) / fx * z, (ys + .5 - cy) / fy * z, z], 1)
        obj = pc @ np.linalg.inv(gt[i])[:3, :3].T + np.linalg.inv(gt[i])[:3, 3]
        vox = set(map(tuple, np.floor(obj / voxel).astype(np.int64)))
        new = vox - seen
        seen |= vox
        new_per_frame.append(len(new))
        total_per_frame.append(len(vox))
    return new_per_frame, total_per_frame


def run_tracker(render_dir, checkpoint, config, overrides):
    """One tracker run; records model size / labels / pose error per frame."""
    from examples.multi_part.naive import NaiveConfig
    from examples.multi_part.acquire import apply_config, apply_overrides
    from examples.multi_part.author_demo import make_tracker
    from examples.multi_part.author_state import union_mask

    render_dir = Path(render_dir)
    meta = json.loads((render_dir / 'meta.json').read_text())
    K = np.array(meta['intrinsics'], float)
    gt = np.load(render_dir / 'poses.npz')['T_cam_part'][:, 0]
    names = sorted(p.name for p in (render_dir / 'rgb').glob('*.png'))
    cfg = apply_config(NaiveConfig(), config)
    apply_overrides(cfg, overrides)
    stream = make_tracker(K, cfg, checkpoint)

    import io
    import re
    import time
    from contextlib import redirect_stdout
    rows, rec, gate_events = [], [], []
    for i, name in enumerate(names):
        rgb = cv2.cvtColor(cv2.imread(str(render_dir / 'rgb' / name)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(render_dir / 'depth' / name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
        mask = union_mask(cv2.imread(str(render_dir / 'seg' / name), cv2.IMREAD_GRAYSCALE))
        t0 = time.perf_counter()
        buf = io.StringIO()
        with redirect_stdout(buf):          # capture the tracker's gate log lines
            if i == 0:
                stream.start(rgb, depth, mask)
                stream.root = 0
            else:
                stream.step(rgb, depth, mask)
        step_ms = 1000 * (time.perf_counter() - t0)
        for line in buf.getvalue().splitlines():
            print(line)                     # keep the log visible
            if line.startswith('[reprojection]'):
                ev = dict(frame=i, raw=line)
                for key, pat, cast in (('gain', r'gain=([-\d.]+)', float),
                                       ('accept', r'accept=(\w+)', lambda s: s == 'True'),
                                       ('mode', r'mode=(\S+)', str),
                                       ('geometry', r'geometry=(\S+)', str),
                                       ('reason', r'reason=(\S+)', str)):
                    m = re.search(pat, line)
                    if m:
                        ev[key] = cast(m.group(1))
                gate_events.append(ev)
        root = stream.parts[stream.root]
        rec.append(root.pose.copy())
        model = getattr(stream, 'model', None)
        n_total = len(model.cloud) if model is not None else 0
        labels = np.asarray(model.labels) if model is not None else np.zeros(0)
        rows.append(dict(frame=i, observed=bool(root.observed),
                         parts=len(stream.parts), n_gauss=n_total,
                         n_assigned=int((labels >= 0).sum()),
                         n_unassigned=int((labels < 0).sum()),
                         step_ms=step_ms))
    rec = np.stack(rec)
    G0, R0 = np.linalg.inv(gt[0]), np.linalg.inv(rec[0])
    for i, r in enumerate(rows):
        E = np.linalg.inv(gt[i] @ G0) @ (rec[i] @ R0)
        r['rot_err_deg'] = float(np.degrees(np.arccos(np.clip((np.trace(E[:3, :3]) - 1) / 2, -1, 1))))
        r['trans_err_m'] = float(np.linalg.norm(E[:3, 3]))
    return rows, gate_events


def summarize(rows, new_vox):
    n = [r['n_gauss'] for r in rows]
    una = [r['n_unassigned'] for r in rows]
    rot = np.array([r['rot_err_deg'] for r in rows])
    tr = np.array([r['trans_err_m'] for r in rows])
    obs = np.array([r['observed'] for r in rows])
    ms = np.array([r['step_ms'] for r in rows[1:]])   # skip start() frame
    growth = int(n[-1] - n[0])

    def cond(sel):
        if not sel.any():
            return dict(n=0)
        return dict(n=int(sel.sum()),
                    rot_med_deg=float(np.median(rot[sel])), rot_max_deg=float(rot[sel].max()),
                    trans_med_m=float(np.median(tr[sel])), trans_max_m=float(tr[sel].max()))

    return dict(
        n_gauss_start=n[0], n_gauss_end=n[-1], cumulative_growth=growth,
        growth_after_frame0_pct=round(100.0 * growth / max(1, n[0]), 1),
        unassigned_end=una[-1], unassigned_max=max(una),
        parts_max=max(r['parts'] for r in rows),
        observed_coverage=float(obs.mean()),
        rot_err_deg=dict(median=float(np.median(rot)), max=float(rot.max())),
        trans_err_m=dict(median=float(np.median(tr)), max=float(tr.max())),
        err_given_observed=cond(obs), err_given_held=cond(~obs),
        step_ms=dict(median=float(np.median(ms)), p90=float(np.percentile(ms, 90))),
        gt_new_voxels_after_frame0=int(sum(new_vox[1:])),
        gt_new_voxels_frame0=int(new_vox[0]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('render_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/tapir/causal_bootstapir_checkpoint.pt')
    ap.add_argument('--config', default='reprojection_split.yaml')
    ap.add_argument('--voxel', type=float, default=0.005)
    ap.add_argument('--run', action='append', default=[], metavar='NAME=CONFIG[:OV;OV]',
                    help="named run, e.g. frozen=reprojection_split_frozen.yaml:dense=true; "
                         "default runs are baseline and growth_off on --config")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    new_vox, tot_vox = gt_new_voxels(a.render_dir, a.voxel)
    if a.run:
        configs = {}
        for spec in a.run:
            name, rest = spec.split('=', 1)
            cfg_name, _, ovs = rest.partition(':')
            configs[name] = (cfg_name, [o for o in ovs.split(';') if o])
    else:
        configs = dict(baseline=(a.config, ['dense=true']),
                       growth_off=(a.config, ['dense=true', 'grow_every=0']))
    result = dict(render_dir=str(a.render_dir), voxel_m=a.voxel,
                  gt_note='GT poses/voxels are EVAL-ONLY; tracker input is rgb, '
                          'depth and the oracle binary union mask (seg != 255)',
                  gt_new_voxels_per_frame=new_vox,
                  gt_total_voxels_per_frame=tot_vox, runs={})
    for name, (cfg_name, ov) in configs.items():
        print(f'[probe] running {name}: {cfg_name} --set ' + ' --set '.join(ov), flush=True)
        rows, gate = run_tracker(a.render_dir, a.checkpoint, cfg_name, ov)
        summary = summarize(rows, new_vox)
        summary['gate'] = dict(
            proposals=len(gate),
            accepted=sum(1 for e in gate if e.get('accept')),
            frozen_geometry_used=sum(1 for e in gate if 'frozen' in str(e.get('geometry', ''))),
            reasons={r: sum(1 for e in gate if e.get('reason') == r)
                     for r in {e.get('reason') for e in gate if e.get('reason')}})
        result['runs'][name] = dict(config=cfg_name, overrides=ov,
                                    summary=summary, per_frame=rows,
                                    gate_events=gate)
    (out / 'growth_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    for name, run in result['runs'].items():
        print(f'\n=== {name} ===')
        print(json.dumps(run['summary'], indent=2))
    print(f"\n[probe] GT new-view ceiling after frame 0: {sum(new_vox[1:])} voxels "
          f"({a.voxel*1000:.0f} mm) -- legitimate growth cannot exceed this.")


if __name__ == '__main__':
    main()

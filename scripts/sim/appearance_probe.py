"""Can stored appearance reject the mirror pose? Oracle-labeled diagnostic.

Uses root's examples/multi_part/appearance_evidence.py unmodified: normalized
RGB L1 over depth-supported, z-buffered stored Gaussian centres. This is a
DIAGNOSTIC, not a rasterized loss and not tracking integration; nothing here
feeds back into the tracker.

Protocol, per arm (uniform / distinct faces, identical seed and motion):
  1. Run the unmodified tracker (dense) on the turnover; record its actual
     per-frame poses. The tracker converges to the mirror (y-flip) pose --
     established in doc/thin_turnover_control.md to <=1 mm surface RMS.
  2. FREEZE the dense model's geometry+colors at the last pre-flip frame
     (before the turnover window opens). The frozen model has seen the top
     face and the four sides, never the bottom.
  3. For every later frame score TWO pose hypotheses on the frozen model:
       tracked   = the tracker's actual (mirror-locked) pose, and
       gt_oracle = gt(t) @ inv(gt(0)) @ rec(0)  -- the pose the tracker SHOULD
                   have, i.e. true GT motion applied to the initial anchor
                   pose. GT enters ONLY through this labeled oracle hypothesis;
                   it is not available to any runtime component.
  4. Report support counts, coverage and colour scores per hypothesis.
     Support counts are generally UNEQUAL between hypotheses; the comparison
     is reported with both counts and is not treated as a fair ranking where
     they diverge.

    python -m scripts.sim.appearance_probe <render_dir> --out <dir>
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np

from examples.multi_part.appearance_evidence import (appearance_evidence,
                                                     paired_appearance_evidence)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('render_dir')
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', default='checkpoints/tapir/causal_bootstapir_checkpoint.pt')
    ap.add_argument('--config', default='reprojection_split.yaml')
    ap.add_argument('--freeze-frame', type=int, default=None,
                    help='default: last frame before the turnover window opens')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    from examples.multi_part.naive import NaiveConfig
    from examples.multi_part.acquire import apply_config, apply_overrides
    from examples.multi_part.author_demo import make_tracker
    from examples.multi_part.author_state import union_mask

    render_dir = Path(a.render_dir)
    meta = json.loads((render_dir / 'meta.json').read_text())
    K = np.array(meta['intrinsics'], float)
    gt = np.load(render_dir / 'poses.npz')['T_cam_part'][:, 0]      # oracle only
    names = sorted(p.name for p in (render_dir / 'rgb').glob('*.png'))
    n = len(names)
    freeze = a.freeze_frame
    if freeze is None:
        freeze = int(meta['turnover_window'][0] * (n - 1)) - 1      # pre-flip
    cfg = apply_config(NaiveConfig(), a.config)
    apply_overrides(cfg, ['dense=true'])
    stream = make_tracker(K, cfg, a.checkpoint)

    frozen = None
    rows = []
    frames = []
    for i, name in enumerate(names):
        rgb = cv2.cvtColor(cv2.imread(str(render_dir / 'rgb' / name)), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(str(render_dir / 'depth' / name), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.
        mask = union_mask(cv2.imread(str(render_dir / 'seg' / name), cv2.IMREAD_GRAYSCALE))
        if i == 0:
            stream.start(rgb, depth, mask)
            stream.root = 0
        else:
            stream.step(rgb, depth, mask)
        frames.append((rgb, depth, mask))
        rows.append(dict(frame=i, pose=stream.parts[stream.root].pose.copy(),
                         observed=bool(stream.parts[stream.root].observed)))
        if i == freeze:
            m = stream.model
            frozen = (m.cloud.means.detach().cpu().numpy().copy(),
                      np.clip(m.cloud.colors.detach().cpu().numpy().copy(), 0, 1))
            print(f'[appear] frozen {len(frozen[0])} centres at pre-flip frame {i}', flush=True)
    if frozen is None:
        raise SystemExit(f'freeze frame {freeze} not reached')
    points, colors = frozen

    rec0 = rows[0]['pose']
    per_frame = []
    # Score EVERY frame with the frozen snapshot: frames before the flip window
    # give the good-tracking baseline distribution, frames after it the
    # mirror-locked one. (Pre-freeze frames are static, so the f-freeze model
    # is valid for them.)
    for i in range(n):
        rgb, depth, mask = frames[i]
        tracked = rows[i]['pose']
        gt_oracle = gt[i] @ np.linalg.inv(gt[0]) @ rec0
        ev_t = appearance_evidence(points, colors, tracked, K, rgb, depth, mask)
        ev_g = appearance_evidence(points, colors, gt_oracle, K, rgb, depth, mask)
        # Paired: SAME stored IDs supported under both poses (min 20); positive
        # gain favours the second (oracle) pose. Support counts accompany it.
        pr = paired_appearance_evidence(points, colors, tracked, gt_oracle,
                                        K, rgb, depth, mask)
        per_frame.append(dict(frame=i, observed=rows[i]['observed'],
                              tracked=ev_t, gt_oracle=ev_g, paired=pr,
                              support_ratio=(ev_t['samples'] / ev_g['samples']
                                             if ev_g['samples'] else None)))

    w0 = int(meta['turnover_window'][0] * (n - 1))
    w1 = int(meta['turnover_window'][1] * (n - 1))
    pre = [r for r in per_frame if r['frame'] < w0]
    post = [r for r in per_frame if r['frame'] > w1]
    def agg(rows_, key):
        s = [r[key]['samples'] for r in rows_]
        l1 = [r[key]['mean_rgb_l1'] for r in rows_ if r[key]['mean_rgb_l1'] is not None]
        return dict(samples_median=int(np.median(s)) if s else 0,
                    samples_min=int(min(s)) if s else 0,
                    mean_rgb_l1_median=float(np.median(l1)) if l1 else None,
                    mean_rgb_l1_final=(rows_[-1][key]['mean_rgb_l1'] if rows_ else None))
    def dist(rows_):
        l1 = np.array([r['tracked']['mean_rgb_l1'] for r in rows_
                       if r['tracked']['mean_rgb_l1'] is not None])
        if not len(l1):
            return dict(n=0)
        return dict(n=len(l1), p5=float(np.percentile(l1, 5)),
                    median=float(np.median(l1)), p95=float(np.percentile(l1, 95)),
                    max=float(l1.max()))
    def paired_agg(rows_):
        p = [r['paired'] for r in rows_ if r['paired']['gain'] is not None]
        if not p:
            return dict(frames_with_min_samples=0)
        return dict(frames_with_min_samples=len(p),
                    paired_samples_median=int(np.median([x['paired_samples'] for x in p])),
                    tracked_l1_median=float(np.median([x['first_rgb_l1'] for x in p])),
                    oracle_l1_median=float(np.median([x['second_rgb_l1'] for x in p])),
                    gain_median=float(np.median([x['gain'] for x in p])),
                    gain_frames_favor_oracle=int(sum(1 for x in p if x['gain'] > 0)))
    summary = dict(
        faces=meta.get('faces', 'uniform'), frames=n, freeze_frame=freeze,
        frozen_centres=len(points),
        pre_flip=dict(tracked=agg(pre, 'tracked'), gt_oracle=agg(pre, 'gt_oracle')),
        post_flip=dict(tracked=agg(post, 'tracked'), gt_oracle=agg(post, 'gt_oracle')),
        paired_post_flip=paired_agg(post),
        tracked_l1_distribution=dict(good_pre_flip=dist(pre), bad_post_flip=dist(post)),
        caveats=['gt_oracle is a LABELED ORACLE diagnostic, not a runtime hypothesis',
                 'support counts are unequal between hypotheses; not a fair ranking '
                 'where they diverge',
                 'frozen model has never seen the bottom face; both hypotheses '
                 'project pre-flip surfaces onto post-flip observations',
                 'directional illumination differs across the flip; colour scores '
                 'include that effect'])
    (out / 'appearance_probe.json').write_text(json.dumps(
        dict(summary=summary, per_frame=per_frame), indent=2, allow_nan=False))
    # Raw arrays for offline breakdown (which stored points are supported under
    # which hypothesis) -- diagnostic data only.
    np.savez_compressed(out / 'appearance_probe_arrays.npz',
                        points=points, colors=colors, rec0=rec0,
                        tracked_final=rows[-1]['pose'],
                        gt_oracle_final=gt[-1] @ np.linalg.inv(gt[0]) @ rec0,
                        K=K)
    cv2.imwrite(str(out / 'final_rgb.png'), cv2.cvtColor(frames[-1][0], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out / 'final_depth.png'),
                np.clip(np.rint(frames[-1][1] * 1000), 0, 65535).astype(np.uint16))
    cv2.imwrite(str(out / 'final_mask.png'), frames[-1][2])
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

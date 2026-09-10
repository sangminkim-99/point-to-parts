"""Run causal RGB-D split ablations; GT is consumed only by replay evaluation."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    'guarded_incremental': ['incremental_pose=true', 'incremental_jump_only=true'],
    'incremental': ['incremental_pose=true'],
    'incremental_nogate': ['incremental_pose=true', 'split_reprojection=false'],
    'bounded_refine': ['refine_step_guard=true'],
    'continuity': ['pose_reprojection=true', 'pose_continuity=true'],
    'continuity_low_support': ['pose_reprojection=true', 'pose_continuity=true', 'pose_min_support=0.1'],
    'proposed_min6': ['split_reprojection_min_points=6'],
    'sparse_min6': ['split_reprojection_dense=false', 'split_reprojection_min_points=6'],
    'temporal_pose': ['pose_reprojection=true'],
    'baseline': ['split_reprojection=false'],
    'proposed': [],
    'symmetric_depth': ['split_reprojection_occlusion=false'],
    'sparse_only': ['split_reprojection_dense=false'],
}


def parse_log(text):
    final = re.search(r'\[eval\] (\S+): (\d+) parts vs (\d+) GT, covered (\d+)/(\d+), purity ([\d.]+)%', text)
    if not final:
        raise ValueError('missing completed evaluation')
    poses = []
    for p, gt, mm, deg, n in re.findall(
            r'first-pose-aligned trajectory p(\d+) \(([^)]+)\): ([\d.]+) mm, ([\d.]+) deg median over (\d+) frames', text):
        poses.append(dict(part_id=int(p), gt=gt, translation_mm=float(mm), rotation_deg=float(deg), frames=int(n)))
    return {'case': final[1], 'parts': int(final[2]), 'gt_parts': int(final[3]),
            'covered': int(final[4]), 'purity_percent': float(final[6]),
            'aligned_trajectories': poses,
            'split_summary': next(l for l in text.splitlines() if '[replay] naive:' in l and 'splits' in l),
            'reprojection_rejections': len(re.findall(r'\[reprojection\].*accept=False', text)),
            'pose_rescues': text.count('[pose-reprojection]'),
            'depth_recovery_frames': int(re.search(r'depth-only recovery: (\d+)', text)[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', default=['laptop_orbit', 'laptop_hinge', 'storage_slide'])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['baseline', 'proposed', 'symmetric_depth', 'sparse_only'])
    parser.add_argument('--trace', action='store_true', help='archive all per-frame IDs/poses, including deleted IDs')
    args = parser.parse_args()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(['git', 'diff'], cwd=ROOT, text=True)
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        for variant in args.variants:
            out = args.out / case / variant
            out.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-u', '-m', 'examples.multi_part.replay',
                   '--seq-dir', str(args.data.resolve() / case), '--method', 'naive',
                   '--config', 'reprojection_split.yaml', '--stride', '1', '--view', '0',
                   '--hyp-panel', '0', '--out', str(out.resolve() / 'tracking.mp4')]
            if args.trace:
                cmd += ['--trace-out', str(out.resolve() / 'trace.npz')]
            for override in VARIANTS[variant]:
                cmd += ['--set', override]
            if (out / 'run.json').exists():
                raise RuntimeError(f'{out}: use a new output directory to preserve old runs')
            (out / 'run.json').write_text(json.dumps({'commit': commit, 'command': cmd,
                'variant': variant, 'dataset_meta': json.loads((args.data / case / 'meta.json').read_text())}, indent=2))
            (out / 'working.patch').write_text(diff)
            with (out / 'tracking.log').open('w') as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            row = {'variant': variant, **parse_log((out / 'tracking.log').read_text())}
            rows.append(row)
            (args.out / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()

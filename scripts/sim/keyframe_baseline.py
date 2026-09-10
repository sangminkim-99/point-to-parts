"""Evaluate a classical rigid RGB-D keyframe baseline on locked-object controls."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.multi_part.keyframe_rigid import RGBDKeyframes
from point2pose.io.sources.dataset.sapien_reader import SapienReader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seq-dir', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--previous-only', action='store_true')
    ap.add_argument('--multi-reference', action='store_true')
    ap.add_argument('--backend', choices=['sift', 'lightglue'], default='sift')
    args = ap.parse_args()
    if (args.out / 'summary.json').exists():
        ap.error('choose a new output directory to preserve previous results')
    reader = SapienReader(args.seq_dir)
    if reader.num_objects != 1:
        ap.error('this baseline is only for a locked single rigid motion group')
    tracker = RGBDKeyframes(reader.K, memory=not args.previous_only, backend=args.backend, multi_reference=args.multi_reference)
    poses, observed, refs, counts, times = [], [], [], [], []
    for i in range(len(reader)):
        rgb, depth = reader.get_color(i), reader.get_depth(i)
        mask = (reader.render_part_index_map(i) >= 0).astype(np.uint8)
        start = time.perf_counter()
        T, ok, n, ref = tracker.step(i, rgb, depth, mask)
        times.append((time.perf_counter()-start)*1000)
        poses.append(T); observed.append(ok); refs.append(ref); counts.append(n)
    # Only evaluation reads GT poses.
    truth = np.array([reader.get_gt_pose(i, reader.object_names[0]) for i in range(len(reader))])
    expected = truth @ np.linalg.inv(truth[0])
    errors = np.linalg.inv(expected) @ np.array(poses)
    mm = np.linalg.norm(errors[:, :3, 3], axis=1)*1000
    deg = np.degrees(Rotation.from_matrix(errors[:, :3, :3]).magnitude())
    result = {'case': reader.video_name, 'method': 'previous_only' if args.previous_only else 'keyframes',
              'backend': args.backend, 'multi_reference': args.multi_reference, 'frames': len(poses), 'observed_frames': int(sum(observed)),
              'all_frame_median_mm': float(np.median(mm)), 'all_frame_median_deg': float(np.median(deg)),
              'final_mm': float(mm[-1]), 'final_deg': float(deg[-1]),
              'median_ms': float(np.median(times)), 'keyframes': [k.frame for k in tracker.keyframes],
              'scope': 'feature/3D-RANSAC rigid baseline, not BundleTrack reproduction'}
    import hashlib
    result['implementation_sha256'] = hashlib.sha256((Path(__file__).read_bytes() + (Path(__file__).resolve().parents[2]/'examples/multi_part/keyframe_rigid.py').read_bytes())).hexdigest()
    result['command'] = sys.argv
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out/'trace.npz', poses=poses, observed=observed, references=refs,
                        inliers=counts, gt_poses=truth, errors_mm=mm, errors_deg=deg, times_ms=times)
    (args.out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

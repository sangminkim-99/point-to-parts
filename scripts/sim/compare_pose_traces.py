"""Evaluation-only, identity-aware comparisons on shared RGB-D trace frames."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def summarize(trace, min_votes=8, min_purity=.6):
    """Assign from seed-label votes, never from pose error; align once per ID/GT.

    Votes are an evaluation proxy and cannot resolve incorrect seed labels. Held
    poses are excluded from observed accuracy but counted as missing coverage.
    """
    frames = trace['frames']
    if len(np.unique(frames)) != len(frames) or np.any(np.diff(frames) <= 0):
        raise ValueError('trace frames must be unique and increasing')
    output = {}
    votes = trace['votes']
    totals = votes.sum(axis=-1)
    dominant = votes.argmax(axis=-1)
    confident = (totals >= min_votes) & (votes.max(axis=-1) >= min_purity * totals)
    for g, name in enumerate(trace['gt_names']):
        records, gauges = {}, {}
        duplicate_frames = 0
        for t, frame in enumerate(frames):
            truth = trace['gt_poses'][t, g]
            candidates = np.flatnonzero(
                (dominant[t] == g) & confident[t] & trace['present'][t]
                & trace['observed'][t] & np.isfinite(trace['poses'][t]).all(axis=(1, 2)))
            if not np.isfinite(truth).all() or not len(candidates):
                continue
            duplicate_frames += int(len(candidates) > 1)
            # Select by label evidence, not best pose; tie uses persistent ID.
            k = min(candidates, key=lambda k: (-int(votes[t, k, g]), int(trace['part_ids'][k])))
            identity = int(trace['part_ids'][k])
            pose = trace['poses'][t, k]
            if identity not in gauges:
                gauges[identity] = np.linalg.inv(truth) @ pose
            error = np.linalg.inv(truth @ gauges[identity]) @ pose
            records[int(frame)] = {
                'id': identity,
                'mm': float(np.linalg.norm(error[:3, 3]) * 1000),
                'deg': float(np.degrees(Rotation.from_matrix(error[:3, :3]).magnitude()))}
        ids = [r['id'] for r in records.values()]
        output[str(name)] = {
            'total_frames': len(frames), 'observed_frames': len(records),
            'coverage': len(records) / len(frames) if len(frames) else 0.,
            'first_observed_frame': next(iter(records), None),
            'missing_frames': [int(f) for f in frames if int(f) not in records],
            'persistent_ids': sorted(set(ids)),
            'id_switches_across_observations': sum(a != b for a, b in zip(ids, ids[1:])),
            'duplicate_candidate_frames': duplicate_frames,
            'records': records}
    return output


def compare(traces, min_votes=8, min_purity=.6):
    names = list(traces)
    first = traces[names[0]]
    # A mismatched sequence must not silently become a matched comparison.
    for trace in traces.values():
        if not np.array_equal(trace['frames'], first['frames']) or not np.array_equal(trace['gt_names'], first['gt_names']):
            raise ValueError('comparison requires identical frame indices and GT names')
        if not np.allclose(trace['gt_poses'], first['gt_poses'], equal_nan=True):
            raise ValueError('comparison requires identical GT trajectories')
    reports = {name: summarize(t, min_votes, min_purity) for name, t in traces.items()}
    output = {'min_votes': min_votes, 'min_purity': min_purity, 'parts': {}}
    for part in reports[names[0]]:
        common = sorted(set.intersection(*(set(r[part]['records']) for r in reports.values())))
        rows = {}
        for name, report in reports.items():
            row = dict(report[part])
            records = row.pop('records')
            for label, selected in [('own_observed', list(records)), ('common', common)]:
                row[label + '_median_mm'] = float(np.median([records[f]['mm'] for f in selected])) if selected else None
                row[label + '_median_deg'] = float(np.median([records[f]['deg'] for f in selected])) if selected else None
            rows[name] = row
        output['parts'][part] = {'common_frames': common, 'runs': rows}
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--trace', action='append', required=True, help='NAME=trace.npz (repeat)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--min-votes', type=int, default=8)
    ap.add_argument('--min-purity', type=float, default=.6)
    args = ap.parse_args()
    if args.min_votes < 1 or not .5 < args.min_purity <= 1:
        ap.error('require min-votes >= 1 and .5 < min-purity <= 1')
    traces, sources = {}, {}
    for item in args.trace:
        name, path = item.split('=', 1)
        if name in traces:
            ap.error('trace names must be unique')
        with np.load(path, allow_pickle=False) as data:
            traces[name] = dict(data)
        sources[name] = str(Path(path).resolve())
    result = compare(traces, args.min_votes, args.min_purity)
    result['sources'] = sources
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()

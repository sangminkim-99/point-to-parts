"""Compare SAPIEN-loaded URDF kinematics with standard URDF matrix composition."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import sapien

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.multi_part.urdf_view import read_urdf, link_poses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('urdf')
    parser.add_argument('--steps', type=int, default=11)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error('steps must be at least two')
    scene = sapien.Scene()
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    art = loader.load(args.urdf)
    links, joints, root = read_urdf(args.urdf)
    active = art.get_active_joints()
    loaded = {link.get_name() for link in art.get_links()}
    if loaded != set(links):
        raise RuntimeError('SAPIEN link set differs from exported URDF')
    worst = 0.
    for fraction in np.linspace(0, 1, args.steps):
        q = [float(j.get_limits()[0, 0] + fraction * np.ptp(j.get_limits()[0])) for j in active]
        art.set_qpos(q)
        values = dict(zip([j.get_name() for j in active], q))
        expected = link_poses(joints, root, [values.get(j['name'], 0.) for j in joints])
        for link in art.get_links():
            actual = link.get_entity_pose().to_transformation_matrix()
            worst = max(worst, float(np.max(np.abs(actual - expected[link.get_name()]))))
    if worst > 2e-6:
        raise AssertionError(f'SAPIEN FK mismatch: {worst}')
    result = {'urdf': str(Path(args.urdf).resolve()), 'steps': args.steps,
              'links': sorted(links), 'active_joints': [j.get_name() for j in active],
              'max_matrix_error': worst, 'passed': True,
              'scope': 'URDF loading and kinematics consistency, not tracking accuracy'}
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()

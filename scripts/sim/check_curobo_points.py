"""GPU smoke test: actual cuRobo Franka FK and mesh-free point/sphere queries."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.manipulation.point_sphere import PartCloud, curobo_clearance


def main():
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cloud', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    model = CudaRobotModel(CudaRobotModelConfig.from_robot_yaml_file('franka.yml'))
    q = model.retract_config.unsqueeze(0).detach().requires_grad_(True)
    spheres = model.get_state(q).link_spheres_tensor
    index = int(torch.nonzero(spheres[0, :, 3] > 0)[-1])
    center = spheres[0, index, :3].detach().clone()
    with np.load(args.cloud, allow_pickle=False) as data:
        points = torch.as_tensor(data['means'][data['labels'] == 0], device='cuda', dtype=q.dtype)
    # Deliberately synthetic placement: put a measured point 1 mm off one robot
    # sphere center. This is a collision fixture, not hand-eye calibration.
    T = torch.eye(4, device='cuda', dtype=q.dtype)
    T[:3, 3] = center - points[0] + points.new_tensor([.001, 0, 0])
    part = PartCloud(0, points, T, .005)
    clearance = curobo_clearance(model, q, [part])
    minimum = clearance.min()
    minimum.backward()
    assert minimum.item() < 0 and torch.isfinite(q.grad).all()
    assert q.grad.norm().item() > 1e-6
    far = T.clone(); far[0, 3] += 10.
    assert curobo_clearance(model, q.detach(), [PartCloud(0, points, far)]).min().item() > 0
    times = []
    with torch.no_grad():
        for _ in range(12):
            torch.cuda.synchronize(); start = time.perf_counter()
            curobo_clearance(model, q.detach(), [part])
            torch.cuda.synchronize(); times.append((time.perf_counter()-start)*1000)
    result = dict(scope='cuRobo FK + custom point/sphere query; synthetic object placement; no planner or physical robot',
        robot='franka.yml', spheres=int(spheres.shape[1]), points=len(points),
        colliding_clearance_m=float(minimum.detach()), finite_joint_gradient=True,
        joint_gradient_norm=float(q.grad.norm()),
        median_query_ms=float(np.median(times[2:])), source_cloud=str(args.cloud))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

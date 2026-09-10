import torch
import pytest
from examples.manipulation.point_sphere import PartCloud, point_sphere_clearance


def test_articulated_frame_and_radius_padding():
    T = torch.eye(4, dtype=torch.float64)
    T[:3, :3] = torch.tensor([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    T[:3, 3] = torch.tensor([1., 2., 3.])
    part = PartCloud(7, torch.tensor([[1., 0, 0]], dtype=T.dtype), T, .02)
    spheres = torch.tensor([[[1., 3., 3.2, .1]]], dtype=T.dtype, requires_grad=True)
    d = point_sphere_clearance(spheres, [part])
    assert d.shape == (1, 1, 1)
    assert d.item() == pytest.approx(.08)
    d.sum().backward()
    assert spheres.grad[0, 0, 2].item() == pytest.approx(1.)


def test_chunking_collision_disabled_spheres_and_empty_geometry():
    pts = torch.tensor([[0., 0, 0], [2., 0, 0], [3., 0, 0]])
    parts = [PartCloud(0, pts, torch.eye(4), 0.), PartCloud(9, pts[:0], torch.eye(4))]
    spheres = torch.tensor([[.05, 0, 0, .1], [0, 0, 0, -1.]])
    d = point_sphere_clearance(spheres, parts, 1)
    assert d[0, 0].item() == pytest.approx(-.05)
    assert torch.isinf(d[1]).all() and torch.isinf(d[:, 1]).all()
    torch.testing.assert_close(d, point_sphere_clearance(spheres, parts, 20))


def test_duplicate_ids_rejected():
    p = PartCloud(0, torch.zeros((1, 3)), torch.eye(4))
    with pytest.raises(ValueError, match='unique'):
        point_sphere_clearance(torch.ones((1, 4)), [p, p])


def test_sweep_catches_thin_surface_missed_by_endpoints():
    from examples.manipulation.point_sphere import swept_point_sphere_clearance
    part = PartCloud(0, torch.zeros((1, 3)), torch.eye(4), 0.)
    spheres = torch.tensor([[[[-1., 0, 0, .1]], [[1., 0, 0, .1]]]])
    assert (point_sphere_clearance(spheres, [part]) > 0).all()
    assert swept_point_sphere_clearance(spheres, [part]).item() == pytest.approx(-.1)


def test_stationary_sweep_and_varying_radius_are_conservative():
    from examples.manipulation.point_sphere import swept_point_sphere_clearance
    part = PartCloud(0, torch.zeros((1, 3)), torch.eye(4), .02)
    spheres = torch.tensor([[[[0., 0, 1., .1]], [[0., 0, 1., .2]]]], requires_grad=True)
    gap = swept_point_sphere_clearance(spheres, [part])
    assert gap.item() == pytest.approx(.78)
    gap.sum().backward()
    assert torch.isfinite(spheres.grad).all()


def test_sweep_rejects_changing_sphere_activation():
    from examples.manipulation.point_sphere import swept_point_sphere_clearance
    spheres = torch.tensor([[[0., 0, 0, .1]], [[0., 0, 0, -1.]]])
    with pytest.raises(ValueError, match='activation'):
        swept_point_sphere_clearance(spheres, [])

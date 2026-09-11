from types import SimpleNamespace as NS
import numpy as np
import torch
from examples.multi_part.render_views import scaled_camera, render_part_views


class Cloud:
    device='cpu'
    def __init__(self):
        self.calls=0
        for key in ('means','colors','scales','quats','opacities'):
            setattr(self,key,torch.zeros((2,3)))
    def render(self, pose, K, h, w, subset):
        self.calls+=1
        return torch.zeros((h,w,3)),torch.ones((h,w)),torch.ones((h,w))


def test_resolution_intrinsics_and_aspect():
    K=np.array([[600,0,320],[0,600,240],[0,0,1.]])
    k,h,w=scaled_camera(K,480,640)
    assert (h,w)==(240,320)
    np.testing.assert_allclose(k,[[300,0,160],[0,300,120],[0,0,1]])
    assert K[0,0]==600


def test_reuse_and_invalidation_for_pose_labels_and_geometry():
    model=NS(cloud=Cloud(),labels=np.array([0,0]))
    parts=[NS(part_id=8,pose=np.eye(4))]
    def get():return render_part_views(model,parts,np.eye(3),48,64,5)
    a,reused=get();assert not reused
    b,reused=get();assert reused and a is b and model.cloud.calls==1
    parts[0].pose[0,3]=.01
    assert not get()[1] and model.cloud.calls==2
    model.labels[1]=-1
    assert not get()[1] and model.cloud.calls==3
    model.cloud.means.add_(1)
    assert not get()[1] and model.cloud.calls==4


def test_rgb_error_uses_only_depth_supported_object_pixels():
    from examples.multi_part.render_views import rendered_rgb_error
    rgb=np.zeros((1,3,3)); obs=np.full((1,3,3),255,np.uint8)
    value,count=rendered_rgb_error(rgb,obs,np.array([[1.,2.,1.]]),np.ones((1,3)),np.array([[1,1,0]]),np.ones((1,3),bool))
    assert value==1. and count==1
    value,count=rendered_rgb_error(rgb,obs,np.ones((1,3)),np.ones((1,3)),np.zeros((1,3)),np.ones((1,3),bool))
    assert value is None and count==0

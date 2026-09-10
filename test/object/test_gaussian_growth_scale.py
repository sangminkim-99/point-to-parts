import numpy as np
import torch
from point2pose.pipeline.components.gaussian_part_assignment import GaussianCloud, GaussianPartAssignment


def test_growing_flat_surface_matches_initial_sample_footprint():
    rgb=np.full((40,40,3),128,np.uint8)
    depth=np.ones((40,40),np.float32)
    K=np.array([[100.,0,20],[0,100.,20],[0,0,1]])
    initial_mask=np.zeros((40,40),np.uint8); initial_mask[12:28,12:28]=1
    cloud=GaussianCloud.from_depth(rgb,depth,K,initial_mask,stride=2,device='cpu')
    before=len(cloud)
    assign=GaussianPartAssignment(cloud,2,device='cpu')
    added,owner=assign.grow_parts(rgb,depth,K,np.ones_like(initial_mask),[np.eye(4)],
        [np.ones(before)],stride=2,dedup_vox=0,confirm=1,max_new=2000)
    assert added>0 and len(owner)==added
    torch.testing.assert_close(cloud.scales[before:],cloud.scales[0].expand(added,3))

import cv2
import numpy as np
from examples.multi_part.recording import Recording


def test_recording_masks_decode_once_and_preserve_frame_lookup(tmp_path):
    (tmp_path/'rgb').mkdir();(tmp_path/'depth').mkdir()
    cv2.imwrite(str(tmp_path/'rgb/000000.png'),np.zeros((3,4,3),np.uint8))
    cv2.imwrite(str(tmp_path/'depth/000000.png'),np.full((3,4),750,np.uint16))
    np.savetxt(tmp_path/'cam_K.txt',np.eye(3))
    masks=np.stack([np.zeros((3,4),np.uint8),np.ones((3,4),np.uint8)])
    np.savez_compressed(tmp_path/'masks.npz',frames=[0,4],masks=masks)
    r=Recording(tmp_path)
    (tmp_path/'masks.npz').unlink()  # no further archive I/O needed for frame queries
    np.testing.assert_array_equal(r.get_mask(4),masks[1])
    assert r.get_mask(1) is None
    np.testing.assert_allclose(r.get_depth(0),.75)

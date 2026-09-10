"""Shared low-resolution Gaussian render products for scoring and display."""
import numpy as np


def scaled_camera(K, height, width, max_width=320):
    w = min(width, max_width)
    h = max(1, round(height * w / width))
    k = np.asarray(K, dtype=np.float32).copy()
    k[0] *= w / width
    k[1] *= h / height
    return k, h, w


def render_part_views(model, parts, K, height, width, frame, max_width=320):
    """Return cached per-part RGB/depth/alpha at the exact model/pose snapshot.

    Consumers can score these products and show them without another rasterizer
    call. Only a single snapshot is retained; candidate pose changes invalidate
    it. Frame IDs alone are insufficient since labels/poses may change in-frame.
    """
    import torch
    k, h, w = scaled_camera(K, height, width, max_width)
    cloud = model.cloud
    labels = np.asarray(model.labels)
    buffers = tuple((id(x), x._version) for x in
                    (cloud.means, cloud.colors, cloud.scales, cloud.quats, cloud.opacities))
    key = (int(frame), h, w, k.tobytes(), labels.tobytes(), buffers,
           tuple((p.part_id, np.asarray(p.pose).tobytes()) for p in parts))
    old = getattr(model, '_render_views_cache', None)
    if old is not None and old['key'] == key:
        return old, True
    views = []
    with torch.no_grad():
        for j, p in enumerate(parts):
            ids = np.flatnonzero(labels == j)
            if not len(ids):
                continue
            rgb, depth, alpha = cloud.render(p.pose, k, h, w,
                                            subset=torch.as_tensor(ids, device=cloud.device))
            views.append(dict(part_id=p.part_id, rgb=rgb.detach().cpu().numpy(),
                              depth=depth.detach().cpu().numpy(), alpha=alpha.detach().cpu().numpy()))
    packet = dict(key=key, frame=int(frame), K=k, height=h, width=w, views=views)
    model._render_views_cache = packet
    return packet, False

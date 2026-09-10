"""Turn a SAPIEN render dir into a tracker-ready Recording (binary union masks).

A render from ``scripts/sim/render_partnet_sequence.py`` holds rgb/, depth/,
seg/ (a uint8 part index, 255 = background) and meta.json (intrinsics). The
online tracker reads a ``Recording``: rgb/, depth/, cam_K.txt and a per-frame
binary mask. This copies rgb/depth through and derives ONLY the binary UNION
mask from seg -- no per-part labels, no GT poses -- providing an oracle binary object mask for simulation controls. This does
not measure live SAM2 segmentation quality.

    python -m examples.multi_part.prep_demo_input <render_dir> <out_dir>
"""
import argparse
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from examples.multi_part.author_state import union_mask


def prepare(render_dir, out_dir):
    render_dir, out_dir = Path(render_dir), Path(out_dir)
    meta = json.loads((render_dir / 'meta.json').read_text())
    K = np.array(meta['intrinsics'], float)
    for sub in ('rgb', 'depth', 'masks'):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    np.savetxt(out_dir / 'cam_K.txt', K)
    names = sorted(p.name for p in (render_dir / 'rgb').glob('*.png'))
    for name in names:
        shutil.copyfile(render_dir / 'rgb' / name, out_dir / 'rgb' / name)
        shutil.copyfile(render_dir / 'depth' / name, out_dir / 'depth' / name)
        seg = cv2.imread(str(render_dir / 'seg' / name), cv2.IMREAD_GRAYSCALE)
        if not cv2.imwrite(str(out_dir / 'masks' / name), union_mask(seg)):
            raise IOError(f'could not write mask {name}')
    # Record that these masks are an ORACLE, so no downstream reader mistakes
    # this for a live-segmentation (SAM2) control.
    provenance = dict(
        mask_source='oracle_sim_seg_union',
        rule='seg != 255 (renderer part index; 255 = background)',
        render_dir=str(render_dir),
        holds_out=['per_part_labels', 'gt_poses'],
        note='Perfect object mask, cleaner than a live SAM2 mask; isolates '
             'tracking from segmentation error and does NOT evaluate SAM2 quality.',
        frames=len(names))
    (out_dir / 'mask_provenance.json').write_text(json.dumps(provenance, indent=2))
    print(f'[prep] {len(names)} frames -> {out_dir} '
          f'(rgb/depth/masks/cam_K.txt + mask_provenance.json; oracle union mask)')
    return len(names)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('render_dir')
    ap.add_argument('out_dir')
    prepare(**vars(ap.parse_args()))


if __name__ == '__main__':
    main()

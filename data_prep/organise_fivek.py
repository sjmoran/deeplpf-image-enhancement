#!/usr/bin/env python
"""Turn raw Lightroom exports into the layout DeepLPF's data loader expects.

Takes the folder of exported INPUT images and the folder of exported Expert-C
TARGET images and writes a clean dataset directory:

    OUT_DIR/
      input/   <id>-<name>.png   (RGB, long edge <= --long-edge)
      output/  <id>-<name>.png

Each image is converted to RGB (alpha dropped) and, unless --no-resize is given,
downscaled so its long edge matches --long-edge (default 512, matching the
DeepLPF example images) without upscaling. Filenames are preserved, so the
`<id>-...` prefix the loader keys on is kept. Input and target are matched by
id (the filename prefix before the first '-'); an image present in only one of
the two exports is skipped with a warning.

Usage:
    python data_prep/organise_fivek.py INPUT_EXPORT_DIR TARGET_EXPORT_DIR OUT_DIR
                                        [--long-edge 512] [--no-resize] [--symlink]
"""
import argparse
import os
import sys

from PIL import Image

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def index_by_id(folder):
    out = {}
    for fn in sorted(os.listdir(folder)):
        if fn.lower().endswith(IMG_EXT):
            out.setdefault(fn.split("-")[0], fn)  # first file per id wins
    return out


def process(src_path, dst_path, long_edge, resize):
    img = Image.open(src_path).convert("RGB")
    if resize:
        w, h = img.size
        scale = long_edge / float(max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    img.save(dst_path)


def main():
    ap = argparse.ArgumentParser(description="Organise Lightroom exports for DeepLPF.")
    ap.add_argument("input_dir", help="folder of exported INPUT images")
    ap.add_argument("target_dir", help="folder of exported Expert-C TARGET images")
    ap.add_argument("out_dir", help="destination dataset directory")
    ap.add_argument("--long-edge", type=int, default=512, help="resize long edge to this (default 512)")
    ap.add_argument("--no-resize", action="store_true", help="keep original resolution")
    args = ap.parse_args()

    inputs = index_by_id(args.input_dir)
    targets = index_by_id(args.target_dir)
    common = sorted(set(inputs) & set(targets))
    only_in = sorted(set(inputs) - set(targets))
    only_out = sorted(set(targets) - set(inputs))

    print(f"inputs: {len(inputs)} | targets: {len(targets)} | paired: {len(common)}")
    if only_in:
        print(f"  {len(only_in)} ids with input but no target (skipped): {only_in[:8]}{' ...' if len(only_in) > 8 else ''}")
    if only_out:
        print(f"  {len(only_out)} ids with target but no input (skipped): {only_out[:8]}{' ...' if len(only_out) > 8 else ''}")

    in_out = os.path.join(args.out_dir, "input")
    tg_out = os.path.join(args.out_dir, "output")
    os.makedirs(in_out, exist_ok=True)
    os.makedirs(tg_out, exist_ok=True)

    resize = not args.no_resize
    for i, iid in enumerate(common, 1):
        in_name = os.path.splitext(inputs[iid])[0] + ".png"
        tg_name = os.path.splitext(targets[iid])[0] + ".png"
        process(os.path.join(args.input_dir, inputs[iid]), os.path.join(in_out, in_name), args.long_edge, resize)
        process(os.path.join(args.target_dir, targets[iid]), os.path.join(tg_out, tg_name), args.long_edge, resize)
        if i % 500 == 0:
            print(f"  processed {i}/{len(common)}")

    print(f"done: wrote {len(common)} pairs to {args.out_dir}/(input|output)")
    print("next: python data_prep/verify_dataset.py", args.out_dir)


if __name__ == "__main__":
    main()

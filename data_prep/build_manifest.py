# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Build a verification manifest for a rendered Adobe-FiveK export.

The FiveK photographs are the photographers' copyright and cannot be
redistributed, so this repository cannot ship the rendered pairs. It can ship a
description of them precise enough that anyone who renders their own export can
check it against ours, and be told what is wrong when it does not match.

For every image the manifest records the SHA-256 of the file, its dimensions,
and per-channel mean and standard deviation. None of that is the photograph.

Exactness versus tolerance: a byte-identical export reproduces the SHA, which is
the strongest check available. Different Lightroom versions, PNG encoders or
colour-management settings can produce a visually identical image with a
different hash, so the statistics are the fallback - they catch the failures
that actually happen (the wrong Inputs rendering, the wrong resize, the wrong
colour space) while tolerating encoder differences.

Usage:
    python3 data_prep/build_manifest.py ./adobe5k_dpe_data adobe5k_dpe/MANIFEST.json.gz
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image


def describe(path):
    """SHA-256, dimensions and per-channel statistics for one image."""
    with open(path, 'rb') as handle:
        raw = handle.read()

    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)

    return {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'bytes': len(raw),
        'height': int(array.shape[0]),
        'width': int(array.shape[1]),
        # Rounded to two decimals: enough to separate a correct render from a
        # wrong one by a wide margin, loose enough not to trip on encoder noise.
        'mean': [round(float(v), 2) for v in array.mean(axis=(0, 1))],
        'std': [round(float(v), 2) for v in array.std(axis=(0, 1))],
    }


def main():
    parser = argparse.ArgumentParser(
        description='Build a verification manifest for a rendered FiveK export')
    parser.add_argument('data_dir', help='directory holding input/ and output/')
    parser.add_argument('out_path', help='where to write the gzipped JSON manifest')
    parser.add_argument('--note', default='',
                        help='free-text note recorded in the manifest header')
    args = parser.parse_args()

    manifest = {
        'description': 'Per-image checksums and statistics for a rendered '
                       'MIT-Adobe FiveK export. The images themselves are the '
                       'photographers\' copyright and are not redistributable; '
                       'this file describes them so an independent export can '
                       'be verified.',
        'rendering': {
            'input': 'Lightroom collection InputAsShotZeroed',
            'target': 'Lightroom collection Experts/C',
            'format': 'PNG, sRGB, 8 bits/component',
            'sizing': 'resize to fit long edge 512 px, do not enlarge',
        },
        'note': args.note,
        'images': {},
    }

    for split in ('input', 'output'):
        directory = os.path.join(args.data_dir, split)
        if not os.path.isdir(directory):
            sys.exit('missing directory: %s' % directory)

        names = sorted(os.listdir(directory))
        manifest['images'][split] = {}

        for index, name in enumerate(names, 1):
            if not name.lower().endswith('.png'):
                continue
            manifest['images'][split][name] = describe(os.path.join(directory, name))
            if index % 500 == 0:
                print('  %s: %d/%d' % (split, index, len(names)), flush=True)

        print('%s: %d images' % (split, len(manifest['images'][split])))

    with gzip.open(args.out_path, 'wt', encoding='utf-8') as handle:
        json.dump(manifest, handle, separators=(',', ':'), sort_keys=True)

    size_kb = os.path.getsize(args.out_path) / 1024.0
    print('wrote %s (%.0f KB)' % (args.out_path, size_kb))


if __name__ == '__main__':
    main()

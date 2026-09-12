# -*- coding: utf-8 -*-
#Copyright (C) 2026. Sean Moran. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""Check a rendered FiveK export against the reference manifest, and say what is wrong.

The FiveK photographs cannot be redistributed, so this repository ships a
manifest of checksums and per-image statistics instead (see
``data_prep/build_manifest.py``). This script compares your own Lightroom export
against it.

The point is not to say pass or fail. Every failure mode this benchmark has is
diagnosable from the statistics, and each one has a specific fix:

  systematically darker      -> the wrong Inputs collection, almost always one
                                of the `... minus 1.5` renderings, which apply a
                                -1.5 EV exposure cut
  systematically brighter    -> an Inputs rendering with positive exposure, or
                                Experts/C exported into input/
  hue shifted                -> the wrong white balance rendering
  dimensions wrong           -> the resize was not "long edge 512, don't enlarge"
  input and target identical -> both collections exported to the same folder
  missing or extra ids       -> the collection was filtered, or a stale export
                                is mixed in

Usage:
    python3 data_prep/verify_export.py ./adobe5k_dpe_data adobe5k_dpe/MANIFEST.json.gz
"""

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

# A correct export usually reproduces the SHA exactly. Encoder and Lightroom
# version differences can change bytes without changing the picture, so a
# statistics check backs it up. These bounds are wide enough to absorb that and
# far tighter than any of the real failure modes, which move the mean by tens of
# levels rather than fractions of one.
MEAN_TOL = 1.5      # per-channel mean, 0-255
STD_TOL = 2.0       # per-channel standard deviation, 0-255
REPORT_LIMIT = 8    # example mismatches to print per category


def describe(path):
    with open(path, 'rb') as handle:
        raw = handle.read()
    array = np.asarray(Image.open(path).convert('RGB'), dtype=np.float64)
    return {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'height': int(array.shape[0]),
        'width': int(array.shape[1]),
        'mean': array.mean(axis=(0, 1)),
        'std': array.std(axis=(0, 1)),
    }


def diagnose(deltas):
    """Turn a set of per-channel mean differences into a probable cause."""
    if not deltas:
        return None

    arr = np.array(deltas)                 # (n, 3), yours minus reference
    overall = float(arr.mean())
    per_channel = arr.mean(axis=0)
    # Hue shift: channels disagree with each other more than they agree overall.
    spread = float(per_channel.max() - per_channel.min())

    if overall < -8 and spread < abs(overall):
        return ('your images are %.1f levels darker than the reference: this is '
                'the signature of a `... minus 1.5` Inputs rendering, which '
                'applies a -1.5 EV exposure cut. Re-export from '
                '`InputAsShotZeroed`.' % abs(overall))
    if overall > 8 and spread < abs(overall):
        return ('your images are %.1f levels brighter than the reference. Check '
                'you exported the `InputAsShotZeroed` collection into input/ '
                'and `Experts/C` into output/, and not the other way round.'
                % overall)
    if spread > 6:
        return ('the channels differ unevenly (R %+.1f, G %+.1f, B %+.1f): a '
                'white-balance difference, so probably a different `Inputs` '
                'rendering. Re-export from `InputAsShotZeroed`.'
                % tuple(per_channel))
    return ('the pixels differ from the reference by %.1f levels on average. '
            'Check the colour space is sRGB and the bit depth is 8.' % abs(overall))


def check_split(data_dir, split, reference, report):
    directory = os.path.join(data_dir, split)
    if not os.path.isdir(directory):
        report['fatal'].append('missing directory: %s' % directory)
        return

    have = {n for n in os.listdir(directory) if n.lower().endswith('.png')}
    want = set(reference)

    report['counts'][split] = {'yours': len(have), 'reference': len(want)}
    missing, extra = sorted(want - have), sorted(have - want)
    if missing:
        report['missing'][split] = missing
    if extra:
        report['extra'][split] = extra

    exact = 0
    stat_ok = 0
    size_bad = []
    stat_bad = []
    deltas = []

    for name in sorted(want & have):
        ref = reference[name]
        got = describe(os.path.join(directory, name))

        if got['sha256'] == ref['sha256']:
            exact += 1
            continue

        if (got['height'], got['width']) != (ref['height'], ref['width']):
            size_bad.append((name, '%dx%d' % (got['width'], got['height']),
                             '%dx%d' % (ref['width'], ref['height'])))
            continue

        delta = got['mean'] - np.array(ref['mean'])
        if (np.abs(delta) <= MEAN_TOL).all() and \
           (np.abs(got['std'] - np.array(ref['std'])) <= STD_TOL).all():
            stat_ok += 1
        else:
            stat_bad.append((name, delta))
            deltas.append(delta)

    report['splits'][split] = {
        'exact': exact,
        'within_tolerance': stat_ok,
        'wrong_size': size_bad,
        'wrong_pixels': stat_bad,
        'diagnosis': diagnose(deltas),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify a rendered FiveK export against the reference manifest')
    parser.add_argument('data_dir', help='directory holding input/ and output/')
    parser.add_argument('manifest', help='path to MANIFEST.json.gz')
    args = parser.parse_args()

    with gzip.open(args.manifest, 'rt', encoding='utf-8') as handle:
        manifest = json.load(handle)

    print('Reference rendering:')
    for key, value in manifest['rendering'].items():
        print('  %-8s %s' % (key + ':', value))
    print()

    report = {'splits': {}, 'counts': {}, 'missing': {}, 'extra': {}, 'fatal': []}
    for split in ('input', 'output'):
        check_split(args.data_dir, split, manifest['images'][split], report)

    for message in report['fatal']:
        print('FATAL: %s' % message)
    if report['fatal']:
        sys.exit(2)

    ok = True
    for split, result in report['splits'].items():
        counts = report['counts'][split]
        total = counts['reference']
        matched = result['exact'] + result['within_tolerance']
        print('%s/  %d of %d images match  (%d byte-identical, %d within tolerance)'
              % (split, matched, total, result['exact'], result['within_tolerance']))

        if split in report['missing']:
            ok = False
            names = report['missing'][split]
            print('   %d missing, e.g. %s' % (len(names), ', '.join(names[:4])))
        if split in report['extra']:
            names = report['extra'][split]
            print('   %d files not in the reference, e.g. %s'
                  % (len(names), ', '.join(names[:4])))

        if result['wrong_size']:
            ok = False
            print('   %d wrong size:' % len(result['wrong_size']))
            for name, got, want in result['wrong_size'][:REPORT_LIMIT]:
                print('     %-44s %s (expected %s)' % (name, got, want))
            print('   -> the export must resize to fit a 512 px long edge, '
                  'with "Don\'t Enlarge" ticked.')

        if result['wrong_pixels']:
            ok = False
            print('   %d differ in pixel values:' % len(result['wrong_pixels']))
            for name, delta in result['wrong_pixels'][:REPORT_LIMIT]:
                print('     %-44s mean delta R%+7.2f G%+7.2f B%+7.2f'
                      % ((name,) + tuple(delta)))
            print('   -> %s' % result['diagnosis'])
        print()

    if ok:
        print('EXPORT VERIFIED - this matches the reference rendering.')
        print('Next: python3 data_prep/verify_dataset.py %s' % args.data_dir)
    else:
        print('EXPORT DOES NOT MATCH. Fix the items above and re-run.')
        sys.exit(1)


if __name__ == '__main__':
    main()

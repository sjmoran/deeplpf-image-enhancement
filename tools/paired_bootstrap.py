# -*- coding: utf-8 -*-
"""Paired bootstrap confidence interval for the difference between two arms.

Every arm is evaluated on the same images, so the comparison must be paired.
Image difficulty dominates the variance - a dark, noisy shot scores badly for
every arm - and differencing per image removes it, giving an interval several
times tighter than comparing two independent means.

WHAT THIS INTERVAL COVERS, AND WHAT IT DOES NOT

It covers test-set sampling: "would this ranking hold on a different draw of
498 images from the same distribution?"

It does NOT cover run-to-run variability: initialisation, data order, and the
checkpoint lottery. Those need multiple seeds; a single run is one draw, and no
amount of resampling the test set estimates them. In this repository the
run-to-run term is the larger of the two, so quoting only this interval would
overstate significance - which is roughly how a field accumulates a decade of
0.2 dB improvements that do not replicate.

Report it as: "+0.19 dB, paired test-set CI +/-0.04; run-to-run variability not
estimated and expected to exceed this."

Usage:
    python3 tools/paired_bootstrap.py A/test_per_image.csv B/test_per_image.csv
    python3 tools/paired_bootstrap.py A.csv B.csv --epoch 449 --metric psnr
"""

import argparse
import csv
import sys

import numpy as np


def load(path, metric, epoch=None):
    """Read one arm's per-image scores, keyed by image name.

    :param path: per-image CSV written by metric.Evaluator.evaluate
    :param metric: 'psnr' or 'ssim'
    :param epoch: which evaluation to read; None takes the last one present
    :returns: mapping from image name to score
    :rtype: dict

    """
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row['epoch']), row['image'], float(row[metric])))

    if not rows:
        raise SystemExit('%s contains no rows' % path)

    if epoch is None:
        epoch = max(r[0] for r in rows)

    scores = {name: value for e, name, value in rows if e == epoch}
    if not scores:
        raise SystemExit('%s has no rows for epoch %d' % (path, epoch))
    return scores, epoch


def paired_bootstrap(diffs, iterations=10000, seed=0):
    """Percentile bootstrap over the paired per-image differences.

    Resampling images (not scores independently) is what keeps the pairing
    intact. PSNR is a log of a mean squared error, so its distribution over
    images is skewed; the percentile bootstrap makes no normality assumption.

    :param diffs: per-image differences, arm A minus arm B
    :param iterations: bootstrap resamples
    :param seed: RNG seed, so a reported interval is reproducible
    :returns: (mean, low, high) of the 95% interval
    :rtype: tuple

    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    means = diffs[rng.integers(0, n, size=(iterations, n))].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def main():
    parser = argparse.ArgumentParser(
        description='Paired bootstrap CI for the difference between two arms')
    parser.add_argument('arm_a', help='per-image CSV for arm A')
    parser.add_argument('arm_b', help='per-image CSV for arm B')
    parser.add_argument('--metric', default='psnr', choices=('psnr', 'ssim'))
    parser.add_argument('--epoch', type=int, default=None,
                        help='evaluation epoch to compare; default the last in each file')
    parser.add_argument('--iterations', type=int, default=10000)
    args = parser.parse_args()

    a, epoch_a = load(args.arm_a, args.metric, args.epoch)
    b, epoch_b = load(args.arm_b, args.metric, args.epoch)

    shared = sorted(set(a) & set(b))
    if not shared:
        raise SystemExit('the two files share no image names')
    if len(shared) != len(a) or len(shared) != len(b):
        print('warning: comparing %d shared images (A has %d, B has %d)'
              % (len(shared), len(a), len(b)), file=sys.stderr)

    diffs = np.array([a[n] - b[n] for n in shared])
    mean, low, high = paired_bootstrap(diffs, args.iterations)

    print('A: %s  epoch %d  mean %s %.4f' % (args.arm_a, epoch_a, args.metric,
                                             np.mean([a[n] for n in shared])))
    print('B: %s  epoch %d  mean %s %.4f' % (args.arm_b, epoch_b, args.metric,
                                             np.mean([b[n] for n in shared])))
    print()
    print('paired difference over %d images: %+.4f  95%% CI [%+.4f, %+.4f]'
          % (len(shared), mean, low, high))
    print('A better on %d of %d images (%.1f%%)'
          % (int((diffs > 0).sum()), len(diffs), 100.0 * (diffs > 0).mean()))

    unpaired = np.sqrt(np.var([a[n] for n in shared], ddof=1) / len(shared)
                       + np.var([b[n] for n in shared], ddof=1) / len(shared))
    print('paired SE %.4f vs unpaired SE %.4f (pairing is %.1fx tighter)'
          % (diffs.std(ddof=1) / np.sqrt(len(diffs)), unpaired,
             unpaired / (diffs.std(ddof=1) / np.sqrt(len(diffs)))))
    print()
    if low > 0 or high < 0:
        print('The interval excludes zero: the difference is resolved ON THIS TEST SET.')
    else:
        print('The interval includes zero: the difference is not resolved even')
        print('before accounting for run-to-run variability.')
    print('This interval does NOT cover seed or checkpoint-selection variability.')


if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""The bootstrap must be paired, and must be honest about a null result.

The tool exists to stop a difference being called significant when it is not,
so its own failure modes matter: an interval that is too tight manufactures
significance, and an unpaired interval on correlated data is too *wide*, which
hides real effects. Both are tested here against constructed data whose answer
is known.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.paired_bootstrap import paired_bootstrap, load


def _write(path, scores, epoch=100):
    with open(path, 'w') as handle:
        handle.write('epoch,image,psnr,ssim\n')
        for name, value in scores.items():
            handle.write('%d,%s,%.6f,%.6f\n' % (epoch, name, value, 0.9))


def test_interval_contains_a_known_shift(tmp_path):
    """A constant +0.2 dB offset must be recovered with a tight interval."""
    rng = np.random.default_rng(0)
    diffs = np.full(498, 0.2) + rng.normal(0, 0.01, 498)

    mean, low, high = paired_bootstrap(diffs)

    assert low < 0.2 < high
    assert high - low < 0.01, 'interval implausibly wide for near-constant data'


def test_interval_includes_zero_when_there_is_no_effect(tmp_path):
    """Pure noise must not be reported as a resolved difference."""
    rng = np.random.default_rng(1)
    diffs = rng.normal(0, 1.0, 498)

    _, low, high = paired_bootstrap(diffs)

    assert low < 0 < high, 'a null effect was reported as resolved'


def test_pairing_is_tighter_than_not_pairing():
    """The whole point: shared image difficulty must cancel."""
    rng = np.random.default_rng(2)
    difficulty = rng.normal(20, 3.0, 498)       # per-image, shared by both arms
    a = difficulty + rng.normal(0.2, 0.05, 498)
    b = difficulty + rng.normal(0.0, 0.05, 498)

    paired_se = (a - b).std(ddof=1) / np.sqrt(498)
    unpaired_se = np.sqrt(a.var(ddof=1) / 498 + b.var(ddof=1) / 498)

    assert unpaired_se > 10 * paired_se, \
        'pairing bought almost nothing; the test data is not representative'


def test_end_to_end_on_csv_files(tmp_path):
    """The CLI must run and resolve a difference it should resolve."""
    rng = np.random.default_rng(3)
    names = ['a%04d.png' % i for i in range(498)]
    difficulty = rng.normal(20, 3.0, 498)

    _write(tmp_path / 'a.csv', dict(zip(names, difficulty + 0.3)))
    _write(tmp_path / 'b.csv', dict(zip(names, difficulty)))

    result = subprocess.run(
        [sys.executable, os.path.join(REPO, 'tools', 'paired_bootstrap.py'),
         str(tmp_path / 'a.csv'), str(tmp_path / 'b.csv')],
        capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert '+0.3000' in result.stdout
    assert 'does NOT cover seed' in result.stdout, \
        'the caveat about run-to-run variability must always be printed'


def test_last_epoch_is_used_by_default(tmp_path):
    """Files hold every evaluation; the default must be the most recent."""
    path = tmp_path / 'multi.csv'
    with open(path, 'w') as handle:
        handle.write('epoch,image,psnr,ssim\n')
        handle.write('25,x.png,20.0,0.9\n')
        handle.write('50,x.png,22.0,0.9\n')

    scores, epoch = load(str(path), 'psnr')
    assert epoch == 50 and scores['x.png'] == 22.0

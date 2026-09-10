# -*- coding: utf-8 -*-
"""The ``deeplpf enhance`` command, which is how most people will use this.

``main.py`` needs a dataset directory, a list of image ids and a checkpoint
path; this takes image files. These tests cover the parts that make it usable
on someone else's photographs: a directory argument, a single file, a mixture
of formats, and a file it cannot use failing without abandoning the rest.
"""

import glob
import os

import numpy as np
import pytest
from PIL import Image

import deeplpf_cli

EXAMPLE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'adobe5k_dpe', 'deeplpf_example_test_input',
                       'a4514-kme_0258.png')

needs_weights = pytest.mark.skipif(
    deeplpf_cli.default_checkpoint() is None,
    reason='no bundled checkpoint in this install')


def write_variants(directory):
    """Write one photograph in several formats and modes."""
    source = Image.open(EXAMPLE)
    source.convert('RGB').save(os.path.join(directory, 'colour.png'))
    source.convert('L').save(os.path.join(directory, 'grey.jpg'))
    source.convert('RGBA').save(os.path.join(directory, 'alpha.png'))


@needs_weights
def test_enhance_a_directory_of_mixed_formats(tmp_path):
    """Greyscale, RGBA and JPEG in one directory all come out enhanced."""
    write_variants(str(tmp_path))
    out = tmp_path / 'out'

    status = deeplpf_cli.main(['enhance', str(tmp_path), '--out', str(out),
                               '--device', 'cpu'])

    assert status == 0
    written = sorted(p.name for p in out.iterdir())
    assert written == ['alpha_enhanced.png', 'colour_enhanced.png',
                       'grey_enhanced.png']


@needs_weights
def test_enhance_one_file_keeps_the_image_size(tmp_path):
    """The enhanced image has the dimensions of the one that went in."""
    source = tmp_path / 'photo.png'
    Image.open(EXAMPLE).convert('RGB').save(source)
    out = tmp_path / 'out'

    deeplpf_cli.main(['enhance', str(source), '--out', str(out),
                      '--device', 'cpu'])

    before = Image.open(source).size
    after = Image.open(out / 'photo_enhanced.png').size
    assert before == after


@needs_weights
def test_one_bad_file_does_not_abandon_the_others(tmp_path, capsys):
    """A directory with an unusable image still enhances everything else."""
    Image.open(EXAMPLE).convert('RGB').save(tmp_path / 'good.png')
    # Below the reflection padding's minimum, so the network cannot run on it.
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / 'tiny.png')
    out = tmp_path / 'out'

    status = deeplpf_cli.main(['enhance', str(tmp_path), '--out', str(out),
                               '--device', 'cpu'])

    assert status == 0
    assert [p.name for p in out.iterdir()] == ['good_enhanced.png']
    assert 'tiny.png' in capsys.readouterr().err


def test_no_images_found_is_an_error(tmp_path, capsys):
    """An empty directory reports it rather than exiting as if it worked."""
    status = deeplpf_cli.main(['enhance', str(tmp_path), '--out',
                               str(tmp_path / 'out'), '--device', 'cpu'])

    assert status == 2
    assert 'no images found' in capsys.readouterr().err


@needs_weights
def test_checkpoint_sidecar_selects_the_fixes(tmp_path, capsys):
    """A checkpoint's fixes must come from its sidecar, not from a default.

    No fix adds or removes a parameter, so weights trained with them load into
    an unfixed model without a missing or unexpected key and then compute a
    different image - a silent wrong answer, measured at 0.30 in [0, 1] on the
    v2 checkpoint.
    """
    import fixes

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    v2 = glob.glob(os.path.join(repo, 'pretrained_models',
                                'adobe_dpe_v2_reconstructed', '*.pt'))
    if not v2:
        pytest.skip('the v2 checkpoint is not in this checkout')

    assert fixes.for_checkpoint(v2[0]) == 'blend,ellipse,fusion,ste'

    source = tmp_path / 'photo.png'
    Image.open(EXAMPLE).convert('RGB').save(source)
    deeplpf_cli.main(['enhance', str(source), '--out', str(tmp_path / 'out'),
                      '--device', 'cpu', '--checkpoint', v2[0]])

    assert 'trained with --fixes=blend,ellipse,fusion,ste' in capsys.readouterr().out
    assert fixes.active() == ('blend', 'ellipse', 'fusion', 'ste')

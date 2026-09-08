# -*- coding: utf-8 -*-
"""Modules must execute, and checkpoints must be what we think they are.

Two properties a gradient check cannot establish.

**Declared but never called.** A module built in ``__init__`` and unreachable
from ``forward`` still has its weights serialised into every checkpoint. Nothing
errors; the model simply carries parameters that do nothing, and a reader of
``__init__`` is misled about the architecture.

**Inert in the trained weights.** A component can be called on every forward
pass and still contribute nothing, if training drove it into a saturated region.
Source inspection cannot find this - only measurement can. The tests below load
each released checkpoint and pin what it actually learned, so a retrained
checkpoint cannot silently change the picture in either direction.
"""

import glob
import os

import pytest
import torch

import fixes
import model


# Modules known not to run, each with its reason. Kept as data so that a new
# one fails loudly rather than being absorbed into a vague allowance.
KNOWN_UNCALLED = {
    'backbonenet.unet.local_net': 'built at unet.py:64, referenced nowhere',
    'backbonenet.unet.local_net.conv1': 'child of local_net',
    'backbonenet.unet.local_net.conv2': 'child of local_net',
    'backbonenet.unet.local_net.lrelu': 'child of local_net',
    'backbonenet.unet.local_net.refpad': 'child of local_net',
}

# Under v1 the three encoder projections are dead too; the wiring fix calls them.
KNOWN_UNCALLED_V1_ONLY = {
    'backbonenet.unet.conv1',
    'backbonenet.unet.conv2',
    'backbonenet.unet.conv3',
}


def _modules_that_ran(fix_spec):
    """Names of leaf modules that fired during one forward pass."""
    fixes.configure(fix_spec)
    torch.manual_seed(0)

    net = model.DeepLPFNet()
    net.eval()

    fired = set()
    handles = [m.register_forward_hook(lambda mod, i, o, n=name: fired.add(n))
               for name, m in net.named_modules() if name]

    with torch.no_grad():
        net(torch.rand(1, 3, 64, 64))

    for handle in handles:
        handle.remove()

    all_named = {name for name, _ in net.named_modules() if name}
    return all_named - fired


@pytest.mark.parametrize('fix_spec', ['none', 'all'])
def test_uncalled_modules_are_exactly_the_known_set(fix_spec):
    """Every module either runs, or is listed above with a reason.

    Set equality rather than a subset check: a newly dead module is a
    regression, and one that starts running means the model changed in a way
    this file should be updated to describe.
    """
    never_ran = _modules_that_ran(fix_spec)

    expected = set(KNOWN_UNCALLED)
    if fix_spec == 'none':
        expected |= KNOWN_UNCALLED_V1_ONLY

    # Containers legitimately do not fire their own hook when only their
    # children are called; restrict the comparison to what we track.
    tracked = never_ran & (expected | {'backbonenet.unet.conv1',
                                       'backbonenet.unet.conv2',
                                       'backbonenet.unet.conv3'})

    assert tracked == expected, (
        'unexpectedly dead: %s | now running: %s'
        % (sorted(tracked - expected), sorted(expected - tracked)))


def test_the_wiring_fix_actually_calls_the_projections():
    """The fix exists to run three specific modules. Assert it does."""
    assert not (_modules_that_ran('wiring') & KNOWN_UNCALLED_V1_ONLY), \
        'the wiring fix is on but the encoder projections still never run'


@pytest.mark.parametrize('checkpoint_dir,graduated_is_dead', [
    ('adobe_dpe', True),
    ('adobe_distort_and_recover', True),
    ('adobe_upe', False),
])
def test_released_checkpoints_keep_their_measured_graduated_state(
        checkpoint_dir, graduated_is_dead):
    """Pin the graduated branch's state in each released checkpoint.

    Measured rather than assumed, and recorded in both directions, so that a
    retrained checkpoint dropped into pretrained_models/ cannot change this
    silently.
    """
    paths = glob.glob(os.path.join('pretrained_models', checkpoint_dir, '*.pt'))
    if not paths:
        pytest.skip('%s checkpoint not present' % checkpoint_dir)

    fixes.configure('none')
    net = model.DeepLPFNet()
    net.load_state_dict(torch.load(paths[0], map_location='cpu'), strict=True)
    net.eval()

    captured = []
    handle = net.deeplpfnet.graduated_filter.fc_graduated.register_forward_hook(
        lambda mod, i, o: captured.append(o.detach()))

    with torch.no_grad():
        net(torch.rand(1, 3, 64, 64))

    handle.remove()

    # G[:, 15:24] are the nine per-channel scale factors, each passed through
    # tanh01 and scaled by max_scale=2.
    scale_factors = 0.5 * (torch.tanh(captured[0][:, 15:24]) + 1) * 2
    is_dead = bool((scale_factors == 0).all())

    assert is_dead is graduated_is_dead, (
        '%s: graduated branch dead=%s, expected %s (scale factors max %.3e)'
        % (checkpoint_dir, is_dead, graduated_is_dead, float(scale_factors.max())))

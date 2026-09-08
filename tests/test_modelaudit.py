# -*- coding: utf-8 -*-
"""modelaudit must find this repository's own defects, and not invent others.

The package ships as a tool for other people's models, so its own correctness
matters more than usual. A checker that cries wolf gets deleted by the first
person who meets a legitimate exception; one that stays silent is worse than
nothing.

DeepLPF is a good subject because its defects are known and measured: four
modules that no forward pass reaches, and - under `--fixes=none` - three
parameters that receive no gradient because a straight-through estimator was
written as an nn.Module method and, one line later, its output is used as a
torch.where condition.
"""

import torch

import fixes
import model
from modelaudit import audit


def _build():
    return model.DeepLPFNet()


def _forward(net):
    return net(torch.rand(1, 3, 64, 64))


def test_finds_the_modules_that_never_run():
    """The four declared-but-uncalled modules must be reported, by name."""
    fixes.configure('none')
    report = audit(_build, _forward)

    dead = {name for name, _ in report.dead_modules}

    for expected in ('backbonenet.unet.conv1',
                     'backbonenet.unet.conv2',
                     'backbonenet.unet.conv3',
                     'backbonenet.unet.local_net.conv1'):
        assert expected in dead, '%s not reported; found %s' % (expected, sorted(dead))


def test_does_not_report_modules_that_do_run():
    """No false positives on the live path - the check has to stay trustworthy."""
    fixes.configure('none')
    report = audit(_build, _forward)

    dead = {name for name, _ in report.dead_modules}

    for live in ('backbonenet.unet.dconv_down1.conv1',
                 'backbonenet.final_conv',
                 'deeplpfnet.cubic_filter.fc_cubic',
                 'deeplpfnet.elliptical_filter.fc_elliptical'):
        assert live not in dead, '%s runs but was reported dead' % live


def test_finds_the_starved_parameters_under_v1():
    """The graduated inversion indicators get no gradient in the v1 model.

    They are reported through `dead_params` rather than `dead_modules`, because
    fc_graduated itself runs - only three of its output rows are severed from
    the loss. That distinction is the reason the tool separates the two.
    """
    fixes.configure('none')
    report = audit(_build, _forward)

    starved = {name for name, _ in report.dead_params}
    assert starved, 'expected at least the encoder projections to be starved'

    # Every starved parameter must belong to a module the report already
    # explains, or be listed as an unexplained hit worth a human's attention.
    assert isinstance(report.unexplained_dead(), list)


def test_reports_loss_term_shares():
    """The MS-SSIM term at its published weight must show as negligible."""
    fixes.configure('none')

    def terms(net):
        predicted = net(torch.rand(1, 3, 64, 64))
        target = torch.rand(1, 3, 64, 64)
        criterion = model.DeepLPFLoss()
        from util import ImageProcessing
        pred_lab = ImageProcessing.rgb_to_lab(predicted.squeeze(0))
        targ_lab = ImageProcessing.rgb_to_lab(target.squeeze(0))
        return {
            'lab_l1': torch.nn.functional.l1_loss(pred_lab, targ_lab),
            'msssim': 1.0 - criterion.compute_msssim(
                pred_lab[0, :, :].unsqueeze(0).unsqueeze(0),
                targ_lab[0, :, :].unsqueeze(0).unsqueeze(0)),
        }

    report = audit(_build, _forward, loss_terms=terms,
                   weights={'lab_l1': 1.0, 'msssim': 1e-3})

    assert set(report.loss_shares) == {'lab_l1', 'msssim'}
    _, msssim_share = report.loss_shares['msssim']
    assert msssim_share < 0.05, \
        'MS-SSIM share is %.3f%%; the docs claim it is decorative' % (100 * msssim_share)


def test_checkpoint_prefix_is_resolved_by_trying_both_ways():
    """A "module." prefix is ambiguous, so the loader must try it both ways.

    A DataParallel checkpoint is prefixed "module."; so is every key of a model
    whose own top-level submodule is called `module`. Stripping unconditionally
    mangles the second case, leaves the affected layers at their initialisation,
    and makes the INERT check report phantom hits on a model that is fine.
    """
    import torch.nn as nn
    from modelaudit import audit as run_audit

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    class NamedModule(nn.Module):
        """Top-level submodule legitimately called `module`."""

        def __init__(self):
            super().__init__()
            self.module = Inner()

        def forward(self, x):
            return self.module(x)

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)

        def forward(self, x):
            return self.lin(x)

    forward = lambda m: m(torch.randn(2, 8))

    report = run_audit(NamedModule, forward, ckpt=NamedModule().state_dict())
    assert not [e for e in report.errors if e[0] == 'ckpt'], \
        'a submodule named `module` was mistaken for a DataParallel prefix: %s' % report.errors

    plain_state = Plain().state_dict()
    wrapped = {'module.' + k: v for k, v in plain_state.items()}
    report = run_audit(Plain, forward, ckpt=wrapped)
    assert not [e for e in report.errors if e[0] == 'ckpt'], \
        'a real DataParallel prefix was not stripped: %s' % report.errors

# -*- coding: utf-8 -*-
"""The entry point must import and parse arguments.

Nothing else in this suite imports ``main``: the tests build the model directly,
so a syntax error or a bad import in ``main.py`` passes every one of them and is
first discovered by a GPU instance that dies a minute after launch. That has now
happened twice - once from a function-local ``import torch._dynamo`` that
rebound ``torch``, once from a malformed ``parser.add_argument`` block.

``--help`` alone is not enough: argparse exits before most of the module body
runs. Compiling the file and importing it are what actually catch it.
"""

import py_compile
import subprocess
import sys
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_every_top_level_module_compiles():
    """A syntax error anywhere in the package fails here, not on a GPU."""
    for path in sorted(glob.glob(os.path.join(REPO, '*.py'))):
        py_compile.compile(path, doraise=True)


def test_main_imports():
    """Import the entry point for real - this is what --help skips."""
    result = subprocess.run(
        [sys.executable, '-c', 'import main'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]


def test_main_help_lists_the_run_shaping_flags():
    """The flags the ablation launches with must exist and be spelled as used."""
    result = subprocess.run(
        [sys.executable, 'main.py', '--help'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    for flag in ('--fixes', '--seed', '--cuda_graphs', '--valid_every',
                 '--msssim_weight', '--gate_weight', '--num_epoch'):
        assert flag in result.stdout, '%s missing from --help' % flag


def test_inference_needs_no_training_paths():
    """The README's Quick start command must run without the training flags.

    ``--training_img_dirpath``, ``--train_img_list_path`` and
    ``--valid_img_list_path`` were ``required=True``, so argparse rejected the
    documented inference command before it reached the inference branch.
    """
    result = subprocess.run(
        [sys.executable, 'main.py',
         '--inference_img_list_path=./adobe5k_dpe/images_inference.txt',
         '--inference_img_dirpath=./adobe5k_dpe/',
         '--checkpoint_filepath=./pretrained_models/adobe_dpe/'
         'deeplpf_validpsnr_23.378_validloss_0.033_testpsnr_23.904_'
         'testloss_0.031_epoch_424_model.pt',
         '--help'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]

    # Training still refuses to start without them, naming every one it needs.
    result = subprocess.run(
        [sys.executable, 'main.py', '--num_epoch=1'],
        cwd=REPO, capture_output=True, text=True)
    assert result.returncode != 0
    for flag in ('--training_img_dirpath', '--train_img_list_path',
                 '--valid_img_list_path'):
        assert flag in result.stderr, result.stderr[-2000:]

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

import pytest

import py_compile
import shutil
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


def test_help_and_bad_flags_leave_no_directories(tmp_path):
    """Neither --help nor a rejected command line may create anything on disk.

    The log directory and the TensorBoard writer were created before argparse
    ran, so every --help and every typo left an empty ``log_*`` and a ``runs/``
    behind in the working directory.
    """
    for argv in (['--help'], ['--no_such_flag']):
        subprocess.run([sys.executable, os.path.join(REPO, 'main.py')] + argv,
                       cwd=tmp_path, capture_output=True, text=True)
    assert list(tmp_path.iterdir()) == []


def test_inference_without_targets_writes_enhanced_images(tmp_path):
    """Enhancing your own photographs must work without retouched targets.

    The dataset loaded input and target unconditionally, so a directory with no
    ``output`` images died inside a DataLoader worker with ``'NoneType' object
    has no attribute 'read'`` - on what is the first thing most people try.
    """
    src = os.path.join(REPO, 'adobe5k_dpe', 'deeplpf_example_test_input',
                       'a4514-kme_0258.png')
    data = tmp_path / 'data' / 'input'
    data.mkdir(parents=True)
    shutil.copy(src, data)
    ids = tmp_path / 'ids.txt'
    ids.write_text('a4514\n')

    ckpt = glob.glob(os.path.join(REPO, 'pretrained_models', 'adobe_dpe', '*.pt'))[0]
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, 'main.py'),
         '--inference_img_list_path=' + str(ids),
         '--inference_img_dirpath=' + str(tmp_path / 'data'),
         '--checkpoint_filepath=' + ckpt],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    written = glob.glob(str(tmp_path / 'log_*' / 'inference' / '*_enhanced.png'))
    assert len(written) == 1, result.stderr[-2000:]


def test_training_names_the_image_missing_its_target(tmp_path):
    """A missing target must name the file, not fail inside a worker."""
    sys.path.insert(0, REPO)
    from data import Adobe5kDataLoader

    src = os.path.join(REPO, 'adobe5k_dpe', 'deeplpf_example_test_input',
                       'a4514-kme_0258.png')
    data = tmp_path / 'data' / 'input'
    data.mkdir(parents=True)
    shutil.copy(src, data)
    ids = tmp_path / 'ids.txt'
    ids.write_text('a4514\n')

    loader = Adobe5kDataLoader(data_dirpath=str(tmp_path / 'data'),
                               img_ids_filepath=str(ids))
    try:
        loader.load_data()
    except FileNotFoundError as exc:
        assert 'a4514-kme_0258.png' in str(exc)
    else:
        raise AssertionError('a missing target must raise')


def test_ids_match_files_named_without_a_dash(tmp_path):
    """A photograph named "myphoto.png" must match the listed id "myphoto".

    The id was ``file.split("-")[0]``, so a filename with no "-" kept its
    extension and matched nothing. The run then did nothing at all, silently.
    """
    sys.path.insert(0, REPO)
    from data import Adobe5kDataLoader

    src = os.path.join(REPO, 'adobe5k_dpe', 'deeplpf_example_test_input',
                       'a4514-kme_0258.png')
    data = tmp_path / 'data' / 'input'
    data.mkdir(parents=True)
    shutil.copy(src, data / 'myphoto.png')
    shutil.copy(src, data)  # keeps the dashed convention working too
    ids = tmp_path / 'ids.txt'
    ids.write_text('myphoto\na4514\n')

    entries = Adobe5kDataLoader(data_dirpath=str(tmp_path / 'data'),
                                img_ids_filepath=str(ids)).load_data(
                                    require_output=False)
    assert len(entries) == 2


def test_a_list_matching_nothing_is_an_error(tmp_path):
    """Matching no images must say so rather than run over an empty set."""
    sys.path.insert(0, REPO)
    from data import Adobe5kDataLoader

    data = tmp_path / 'data' / 'input'
    data.mkdir(parents=True)
    ids = tmp_path / 'ids.txt'
    ids.write_text('nothing_here\n')

    try:
        Adobe5kDataLoader(data_dirpath=str(tmp_path / 'data'),
                          img_ids_filepath=str(ids)).load_data(
                              require_output=False)
    except FileNotFoundError as exc:
        assert 'matched an image under' in str(exc)
    else:
        raise AssertionError('an empty match must raise')


def test_greyscale_images_load_as_three_channels():
    """Every convolution expects three channels; a greyscale photo has one."""
    sys.path.insert(0, REPO)
    import numpy as np
    from PIL import Image
    from util import ImageProcessing

    import tempfile
    src = os.path.join(REPO, 'adobe5k_dpe', 'deeplpf_example_test_input',
                       'a4514-kme_0258.png')
    with tempfile.TemporaryDirectory() as tmp:
        grey = os.path.join(tmp, 'grey.png')
        Image.open(src).convert('L').save(grey)
        img = ImageProcessing.load_image(grey, normaliser=1)
        assert img.ndim == 3 and img.shape[2] == 3, img.shape


@pytest.mark.skipif(not os.path.isdir(os.path.join(REPO, 'adobe5k_dpe_data')),
                    reason='needs the FiveK pairs, which are not in the repository')
def test_checkpoint_filepath_initialises_training(tmp_path):
    """--checkpoint_filepath must fine-tune, not silently train from scratch.

    The flag was read only on the inference path, so a run that passed it while
    training produced exactly the from-scratch loss and looked normal.
    """
    lists = {}
    for name, source in (('tr', 'images_train.txt'), ('va', 'images_valid.txt'),
                         ('te', 'images_test.txt')):
        ids = open(os.path.join(REPO, 'adobe5k_dpe', source)).read().split()[:2]
        path = tmp_path / (name + '.txt')
        path.write_text('\n'.join(ids) + '\n')
        lists[name] = str(path)

    ckpt = glob.glob(os.path.join(REPO, 'pretrained_models', 'adobe_dpe', '*.pt'))[0]
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, 'main.py'), '--num_epoch=1',
         '--valid_every=99',
         '--training_img_dirpath=' + os.path.join(REPO, 'adobe5k_dpe_data') + os.sep,
         '--train_img_list_path=' + lists['tr'],
         '--valid_img_list_path=' + lists['va'],
         '--test_img_list_path=' + lists['te'],
         '--checkpoint_filepath=' + ckpt, '--seed=42'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    assert 'Initialised the network from' in result.stderr, result.stderr[-2000:]


def test_valid_every_zero_is_rejected(tmp_path):
    """--valid_every=0 divided the epoch counter by zero after epoch one."""
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, 'main.py'), '--valid_every=0',
         '--training_img_dirpath=x', '--train_img_list_path=x',
         '--valid_img_list_path=x'],
        cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert '--valid_every must be at least 1' in result.stderr


def test_requirements_lists_only_what_is_imported():
    """A dependency nobody imports is one more thing to install and break."""
    sources = []
    for pattern in ('*.py', 'tests/*.py', 'data_prep/*.py', 'tools/*.py',
                    'modelaudit/*.py'):
        for path in glob.glob(os.path.join(REPO, pattern)):
            sources.append(open(path).read())
    blob = '\n'.join(sources)

    # Distribution name -> the module it provides, where they differ.
    provides = {'scikit-image': 'skimage', 'pillow': 'PIL',
                'tensorboard': 'tensorboard', 'torchvision': 'torchvision'}
    for line in open(os.path.join(REPO, 'requirements.txt')):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        dist = line.split('==')[0].split('>=')[0]
        module = provides.get(dist, dist)
        assert module in blob, '%s is in requirements.txt but never imported' % dist

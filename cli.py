# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD-3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""Command-line arguments for training and inference.

Split out of ``main.py``: the flag definitions are most of what a reader
scrolls past, so they sit apart from what the program does with them.
"""
import argparse


def build_parser():
    """Build the argument parser for :func:`main.main`.

    :returns: the parser, with every training and inference flag registered
    :rtype: argparse.ArgumentParser

    """
    parser = argparse.ArgumentParser(
        description="Train the DeepLPF neural network on image pairs")

    parser.add_argument(
        "--num_epoch", type=int, required=False, help="Number of epochs (default 100000)", default=100000)
    parser.add_argument(
        "--batch_size", type=int, required=False, help="Training batch size (default 1)", default=1)
    parser.add_argument(
        "--crop_size", type=int, required=False, default=0,
        help="If >0, randomly crop training images to this square size. Required "
             "for --batch_size>1 because FiveK images vary in size.")
    parser.add_argument(
        "--valid_every", type=int, required=False, help="Number of epochs after which to compute validation accuracy",
        default=25)
    parser.add_argument(
        "--checkpoint_filepath", required=False, help="Location of checkpoint file", default=None)
    parser.add_argument(
        "--inference_img_dirpath", required=False,
        help="Directory containing images to run through a saved DeepLPF model instance", default=None)
    parser.add_argument(
        "--training_img_dirpath", required=False,
        help="Directory containing images to train a DeepLPF model instance", default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument(
        "--inference_img_list_path", required=False,
        help="Plain text file containing the names of the images to inference")
    parser.add_argument(
        "--train_img_list_path", required=False,
        help="Plain text file containing the names of the training images")
    parser.add_argument(
        "--valid_img_list_path", required=False,
        help="Plain text file containing the names of the validation images")
    parser.add_argument(
        "--test_img_list_path", required=False,
        help="Plain text file containing the names of the test images")

    parser.add_argument(
        "--msssim_weight", type=float, required=False, default=None,
        help="Weight on the MS-SSIM term of Eq. 8. Defaults to the published "
             "1e-3, at which the term contributes ~0.07%% of the L1 gradient "
             "and cannot affect training. Raise it to test whether the "
             "structural term matters when it is not decorative.")

    parser.add_argument(
        "--gate_weight", type=float, required=False, default=None,
        help="Weight on the L1 gate penalty of the `gates` feature, which is "
             "what makes the number of active filter instances learned rather "
             "than fixed at three per branch. Too small and every gate pins "
             "at 1 (the published model); too large and the branches collapse "
             "to the identity. Defaults to 3e-3.")

    parser.add_argument(
        "--checkpoint_every", type=int, required=False, default=None,
        help="Also save a checkpoint every N validation rounds regardless of "
             "whether validation improved. Saving only on improvement leaves "
             "late training almost unsampled - runs here produced 125-epoch "
             "gaps - so there is nothing to average over and no way to "
             "estimate run-to-run variability. Costs disk, nothing else.")

    parser.add_argument(
        "--colour_knots", type=int, required=False, default=None,
        help="Knots in the per-channel tone curve of the `colour` feature. "
             "0 leaves the global colour mixer alone, which is the ablation "
             "that says which half carries any gain. Defaults to 16.")

    parser.add_argument(
        "--seed", type=int, required=False, default=None,
        help="Seed for torch, numpy and Python RNGs. Without it every run "
             "starts from a different initialisation and shuffle order, so "
             "arms of an ablation differ by luck as well as by the change "
             "under test. Set it for any comparison between runs.")
    parser.add_argument(
        "--tf32", action="store_true",
        help="Allow TF32 matmul/conv on Ampere+ GPUs. Faster, with a small "
             "loss of mantissa precision. Off by default so the replication "
             "path keeps full fp32.")
    parser.add_argument(
        "--compile", action="store_true", dest="use_compile",
        help="Wrap the network in torch.compile. Fuses kernels, which is "
             "where this model's time actually goes at batch size 1. Costs a "
             "one-off compilation at startup.")
    parser.add_argument(
        "--cuda_graphs", action="store_true",
        help="Capture the whole training step (forward, loss, backward, Adam) "
             "in one CUDA graph per image shape and replay it. Removes the "
             "per-kernel launch cost the model is bound by at batch size 1. "
             "Same kernels and arithmetic as eager; Adam runs capturable, "
             "which computes its bias corrections in fp32 on the device. "
             "Requires CUDA. See trainstep.py.")
    parser.add_argument(
        "--amp", choices=("bf16",), default=None,
        help="Run the network's forward pass under bf16 autocast; the loss "
             "stays fp32. Changes the numerics: against fp32 on an untrained "
             "model, max abs prediction difference 0.012, loss 6e-5, "
             "parameter gradients 0.6-1.3%% of their norm. Off by default.")
    parser.add_argument(
        "--fixes", required=False, default="none",
        help="Which v2 fixes to enable: 'none' (the published v1 model), "
             "'all', or a comma-separated subset of "
             "wiring,ellipse,ste,msssim. Default 'none'.")


    return parser

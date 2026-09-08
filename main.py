# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD-3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
'''
This is a PyTorch implementation of the CVPR 2020 paper:
"Deep Local Parametric Filters for Image Enhancement": https://arxiv.org/abs/2003.13985

Please cite the paper if you use this code

Tested with Pytorch 1.7.1, Python 3.7.9

Authors: Sean Moran (sean.j.moran@gmail.com), 
         Pierre Marza (pierre.marza@gmail.com)

Instructions:

To get this code working on your system / problem please see the README.

BATCH SIZE: training supports a batch size greater than one via --batch_size.
Evaluation and inference run at a batch size of 1 so per-image PSNR/SSIM are
reported and saved individually. To reproduce the paper's reported results,
train with a batch size of 1.
'''
import model
import fixes
import random
import numpy as np
import metric
import os
import datetime
import torch.optim as optim
import argparse
import logging
import torchvision.transforms as transforms
import torch
import torch._dynamo
import trainstep
from data import Adobe5kDataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

def main():
    """Entry point for training and inference.

    Parses command-line arguments and either (a) runs inference with a saved
    checkpoint when both ``--checkpoint_filepath`` and ``--inference_img_dirpath``
    are supplied, or (b) trains a DeepLPF model from scratch on the Adobe5k
    dataset, periodically evaluating on the validation and test splits and
    saving the best-PSNR checkpoint.

    Training supports a batch size greater than one via ``--batch_size``;
    evaluation and inference run at a batch size of 1 so per-image PSNR/SSIM are
    reported and saved individually.
    """

    writer = SummaryWriter()

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dirpath = "./log_" + timestamp
    os.mkdir(log_dirpath)

    handlers = [logging.FileHandler(
        log_dirpath + "/deep_lpf.log"), logging.StreamHandler()]
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s', handlers=handlers)

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
        "--training_img_dirpath", required=True,
        help="Directory containing images to train a DeepLPF model instance", default="/home/sjm213/adobe5k/adobe5k/")
    parser.add_argument(
        "--inference_img_list_path", required=False,
        help="Plain text file containing the names of the images to inference")
    parser.add_argument(
        "--train_img_list_path", required=True,
        help="Plain text file containing the names of the training images")
    parser.add_argument(
        "--valid_img_list_path", required=True,
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
        help="weight on the gates L1 penalty (fixes=...,gates); default 3e-3")    parser.add_argument(
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

    args = parser.parse_args()
    active_fixes = fixes.configure(args.fixes, args.msssim_weight, args.gate_weight)

    if args.seed is not None:
        # Seed every source the training loop draws on: weight init, the
        # DataLoader's shuffle, and the augmentation flips in data.py.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    if args.tf32:
        # Ampere and later: run matmuls and convolutions in TF32.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    num_epoch = args.num_epoch
    batch_size = args.batch_size
    crop_size = args.crop_size
    valid_every = args.valid_every
    checkpoint_filepath = args.checkpoint_filepath
    inference_img_dirpath = args.inference_img_dirpath
    training_img_dirpath = args.training_img_dirpath
    inference_img_list_path = args.inference_img_list_path
    test_img_list_path = args.test_img_list_path
    valid_img_list_path = args.valid_img_list_path
    train_img_list_path = args.train_img_list_path

    logging.info('######### Parameters #########')
    logging.info('Number of epochs: ' + str(num_epoch))
    logging.info('Seed: ' + (str(args.seed) if args.seed is not None else 'unseeded (runs are not comparable)'))
    logging.info('TF32: ' + str(args.tf32) + ', torch.compile: ' + str(args.use_compile)
                 + ', CUDA graphs: ' + str(args.cuda_graphs) + ', amp: ' + str(args.amp))
    logging.info('MS-SSIM weight: ' + str(fixes.msssim_weight()))
    if fixes.enabled('gates'):
        logging.info('Gate penalty weight: ' + str(fixes.gate_weight()))
    logging.info('v2 fixes enabled: ' + (', '.join(active_fixes) or 'none (v1 model)'))
    logging.info('Logging directory: ' + str(log_dirpath))
    logging.info('Dump validation accuracy every: ' + str(valid_every))
    logging.info('Training image directory: ' + str(training_img_dirpath))
    logging.info('List of images to inference: ' + str(inference_img_list_path))
    logging.info('List of test images: ' + str(test_img_list_path))
    logging.info('List of validation images: ' + str(valid_img_list_path))
    logging.info('List of training images: ' + str(train_img_list_path))

    logging.info('##############################')

    # Select the best available device: CUDA GPU, then Apple Silicon (MPS), then CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info('Using device: ' + str(device))

    # Training batch size (configurable). Evaluation and inference always run at
    # a batch size of 1 so per-image PSNR/SSIM are reported and saved.
    BATCH_SIZE = batch_size
    logging.info('Training batch size: ' + str(BATCH_SIZE))

    if (checkpoint_filepath is not None) and (inference_img_dirpath is not None):

        '''
        inference_img_dirpath: the actual filepath should have "input" in the name an in the level above where the images 
        for inference are located, there should be a file "images_inference.txt with each image filename as one line i.e."
        
        images_inference.txt    ../
                                a1000.tif
                                a1242.tif
                                etc
        '''
        inference_data_loader = Adobe5kDataLoader(data_dirpath=inference_img_dirpath,
                                                  img_ids_filepath=inference_img_list_path)
        inference_data_dict = inference_data_loader.load_data()
        inference_dataset = Dataset(data_dict=inference_data_dict,
                                    transform=transforms.Compose([transforms.ToTensor()]), normaliser=1,
                                    is_inference=True)

        inference_data_loader = torch.utils.data.DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                                            num_workers=6)

        '''
        Performs inference on all the images in inference_img_dirpath
        '''
        logging.info(
            "Performing inference with images in directory: " + inference_img_dirpath)

        net = model.DeepLPFNet()
        net.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
        net.to(device)
        net.eval()

        criterion = model.DeepLPFLoss()

        inference_evaluator = metric.Evaluator(
            criterion, inference_data_loader, "test", log_dirpath)

        inference_evaluator.evaluate(net, epoch=0)

    else:

        training_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                                 img_ids_filepath=train_img_list_path)
        training_data_dict = training_data_loader.load_data()

        if BATCH_SIZE > 1 and crop_size <= 0:
            logging.warning(
                "batch_size > 1 requires uniform image sizes; FiveK images vary. "
                "Pass --crop_size (e.g. --crop_size 256) or the data loader will "
                "fail to collate. Continuing anyway.")
        training_dataset = Dataset(data_dict=training_data_dict, normaliser=1, is_valid=False,
                                   crop_size=(crop_size if crop_size > 0 else None))

        validation_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                               img_ids_filepath=valid_img_list_path)
        validation_data_dict = validation_data_loader.load_data()
        validation_dataset = Dataset(data_dict=validation_data_dict, normaliser=1, is_valid=True)

        testing_data_loader = Adobe5kDataLoader(data_dirpath=training_img_dirpath,
                                            img_ids_filepath=test_img_list_path)
        testing_data_dict = testing_data_loader.load_data()
        testing_dataset = Dataset(data_dict=testing_data_dict, normaliser=1,is_valid=True)

        # num_workers is part of the seeded recipe: the per-worker RNG that
        # draws the augmentation flips is seeded from (base_seed + worker_id),
        # so changing the worker count changes which image gets which flip.
        # pin_memory only changes where the host copy lives.
        training_data_loader = torch.utils.data.DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                                       num_workers=6, pin_memory=True)
        # Evaluation runs at batch size 1 so per-image metrics are reported/saved.
        testing_data_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1, shuffle=False,
                                                      num_workers=6)
        validation_data_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1,
                                                         shuffle=False,
                                                         num_workers=6)
        net = model.DeepLPFNet()
        net.to(device)

        logging.info('######### Network created #########')
        logging.info('Architecture:\n' + str(net))

        # Paper Eq. 8 / Sec. 4.1: L1 in CIELab + 1e-3 * (1 - MS-SSIM) on the L channel.
        criterion = model.DeepLPFLoss(ssim_window_size=5)

        '''
        The following objects allow for evaluation of a model on the testing and validation splits of a dataset
        '''
        validation_evaluator = metric.Evaluator(
            criterion, validation_data_loader, "valid", log_dirpath)
        testing_evaluator = metric.Evaluator(
            criterion, testing_data_loader, "test", log_dirpath)

        # Adam with lr 1e-4, as in the paper's implementation details (Sec. 4.1).
        if args.use_compile:
            # Compile after the weights are on the device, and keep the
            # optimiser bound to the original module's parameters.
            # FiveK has 16 distinct image sizes, which dynamo turns into 9
            # graphs (measured with backend='eager'; no graph breaks). The
            # default cache_size_limit is 8, so the ninth shape silently fell
            # back to eager for the rest of the run. Raise it so every shape
            # gets compiled; expect the first epoch to spend minutes compiling.
            # NB: no `import torch._dynamo` here. A function-local import of a
            # torch submodule rebinds `torch` as a local for the WHOLE function,
            # so every earlier `torch.` reference in main() raises
            # UnboundLocalError. It is imported at module scope instead.
            torch._dynamo.config.cache_size_limit = 64
            logging.info('Compiling the network with torch.compile (cache_size_limit=64)')
            net = torch.compile(net)

        if args.cuda_graphs and device.type != 'cuda':
            raise SystemExit('--cuda_graphs needs a CUDA device')
        # capturable=True keeps Adam's step count on the device so the update
        # can be captured; it is the only optimiser change graphs need.
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4, betas=(0.9, 0.999),
                               eps=1e-08, capturable=args.cuda_graphs)
        amp_dtype = torch.bfloat16 if args.amp == 'bf16' else None
        step_cls = trainstep.GraphedStep if args.cuda_graphs else trainstep.EagerStep
        train_step = step_cls(net, criterion, optimizer, autocast_dtype=amp_dtype)
        best_valid_psnr = 0.0

        optimizer.zero_grad()
        net.train()

        total_examples = 0  # running count over all epochs; x-axis of the per-batch loss curve

        for epoch in range(num_epoch):

            # Train loss
            examples = 0.0
            running_loss = 0.0

            for batch_num, data in enumerate(training_data_loader, 0):

                input_img_batch = data['input_img'].to(device, non_blocking=True)
                gt_img_batch = data['output_img'].to(device, non_blocking=True)

                # Forward, loss, backward, Adam: eager, or one graph replay.
                loss = train_step(input_img_batch, gt_img_batch)

                # One device-to-host sync per step, not two.
                loss_value = loss.item()
                running_loss += loss_value
                examples += BATCH_SIZE
                total_examples+=BATCH_SIZE

                writer.add_scalar('Loss/train', loss_value, total_examples)

            logging.info('[%d] train loss: %.15f' %
                         (epoch + 1, running_loss / examples))
            writer.add_scalar('Loss/train_smooth', running_loss / examples, epoch + 1)

            if (epoch + 1) % valid_every == 0:

                logging.info("Evaluating model on validation and test dataset")

                valid_loss, valid_psnr, valid_ssim = validation_evaluator.evaluate(
                    net, epoch)
                test_loss, test_psnr, test_ssim = testing_evaluator.evaluate(
                    net, epoch)

                # Checkpoint whenever validation PSNR improves (model selection is on the validation split).
                if valid_psnr > best_valid_psnr:

                    # Evaluator.evaluate returns plain floats, so use them
                    # directly. This read valid_loss.tolist()[0], which worked
                    # only while running_loss accumulated 1-element tensors.
                    snapshot_name = 'deeplpf_validpsnr_{}_validloss_{}_testpsnr_{}_testloss_{}_epoch_{}_model.pt'.format(
                        valid_psnr, valid_loss, test_psnr, test_loss, epoch)
                    logging.info(
                        "Validation PSNR has increased. Saving the more accurate model to file: " + snapshot_name)

                    best_valid_psnr = valid_psnr
                    torch.save(net.state_dict(), os.path.join(log_dirpath, snapshot_name))

                net.train()

        '''
        Run the network over the testing dataset split
        '''
        testing_evaluator.evaluate(net, epoch=0)

        snapshot_prefix = os.path.join(log_dirpath, 'deep_lpf')
        snapshot_path = snapshot_prefix + "_" + str(num_epoch)
        torch.save(net.state_dict(), snapshot_path)


if __name__ == "__main__":
    main()

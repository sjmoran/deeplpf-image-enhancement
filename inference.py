# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD-3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
"""Running a saved checkpoint over a directory of images."""
import logging

import torch
import torchvision.transforms as transforms

import metric
import model
from data import Adobe5kDataLoader, Dataset


def run_inference(checkpoint_filepath, inference_img_dirpath,
                  inference_img_list_path, device, log_dirpath):
    """Score a saved checkpoint over the images named in a list file.

    ``inference_img_dirpath`` should hold an ``input`` directory, with a text
    file naming one image per line in the level above it.

    :param checkpoint_filepath: state dict to load
    :param inference_img_dirpath: directory holding the images
    :param inference_img_list_path: text file naming the images to run
    :param device: device to run on
    :param log_dirpath: directory for per-image scores and enhanced images
    :returns: N/A

    """

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

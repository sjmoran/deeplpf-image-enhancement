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

'''
import os
import os.path
import torchvision.transforms.functional as TF
import util
import numpy as np
import logging
from collections import defaultdict
import torch
import random
import matplotlib
import sys
from abc import abstractmethod
matplotlib.use('agg')
np.set_printoptions(threshold=sys.maxsize)


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset of (input, target) image pairs with lazy loading.

    Wraps a dictionary of image-pair filepaths and loads each pair from disk on
    demand. During training (``is_valid=False``) it applies random horizontal
    and vertical flips as data augmentation; validation, test and inference
    modes load the images unaugmented.
    """

    def __init__(self, data_dict, transform=None, normaliser=2 ** 8 - 1, is_valid=False, is_inference=False, crop_size=None):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param transform: PyTorch image transformations to apply to the images
        :param crop_size: if set, training images are randomly cropped to this
            square size (the same crop is applied to input and target). Needed
            for a batch size greater than one, since FiveK images vary in size
            and the default collation requires a uniform size. Ignored for
            validation and inference.
        :returns: N/A
        :rtype: N/A

        """
        self.transform = transform
        self.data_dict = data_dict
        self.normaliser = normaliser
        self.is_valid = is_valid
        self.is_inference = is_inference
        self.crop_size = crop_size

    def __len__(self):
        """Returns the number of images in the dataset

        :returns: number of images in the dataset
        :rtype: Integer

        """
        return (len(self.data_dict.keys()))

    def __getitem__(self, idx):
        """Returns a pair of images with the given identifier. This is lazy loading
        of data into memory. Only those image pairs needed for the current batch
        are loaded.

        :param idx: image pair identifier
        :returns: dictionary containing input and output images and their identifier
        :rtype: dictionary

        """
        while True:

            if self.is_inference:

                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)
                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_pil_image(output_img)
                output_img = TF.to_tensor(output_img)

                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}

            elif idx in self.data_dict:

                output_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['output_img'], normaliser=self.normaliser)
                input_img = util.ImageProcessing.load_image(
                    self.data_dict[idx]['input_img'], normaliser=self.normaliser)

                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                    output_img = output_img.astype(np.uint8)

                input_img = TF.to_pil_image(input_img)
                output_img = TF.to_pil_image(output_img)

                if not self.is_valid:

                    if random.random()>0.5:

                        # Random horizontal flipping
                        if random.random() > 0.5:
                            input_img = TF.hflip(input_img)
                            output_img = TF.hflip(output_img)

                        # Random vertical flipping
                        if random.random() > 0.5:
                            input_img = TF.vflip(input_img)
                            output_img = TF.vflip(output_img)

                    # Random square crop (same crop for input and target) so
                    # that a batch size > 1 can collate variable-size images.
                    if self.crop_size:
                        width, height = input_img.size
                        crop = min(self.crop_size, width, height)
                        left = random.randint(0, width - crop)
                        top = random.randint(0, height - crop)
                        box = (left, top, left + crop, top + crop)
                        input_img = input_img.crop(box)
                        output_img = output_img.crop(box)

                # Transform to tensor
                input_img = TF.to_tensor(input_img)
                output_img = TF.to_tensor(output_img)


                return {'input_img': input_img, 'output_img': output_img,
                        'name': self.data_dict[idx]['input_img'].split("/")[-1]}


class DataLoader():
    """Abstract base class for dataset-specific loaders.

    Subclasses implement :meth:`load_data` to scan a directory and build the
    dictionary of image-pair filepaths consumed by :class:`Dataset`.
    """

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        self.data_dirpath = data_dirpath
        self.img_ids_filepath = img_ids_filepath

    @abstractmethod
    def load_data(self):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass

    @abstractmethod
    def perform_inference(self, net, data_dirpath):
        """Abstract function for the data loader class

        :returns: N/A
        :rtype: N/A

        """
        pass


class Adobe5kDataLoader(DataLoader):
    """Data loader for the Adobe5k image-enhancement dataset.

    Walks ``data_dirpath`` for images whose id (the filename prefix before the
    first ``-``) appears in ``img_ids_filepath``, pairing each ``input`` image
    with its corresponding ``output`` image.
    """

    def __init__(self, data_dirpath, img_ids_filepath):
        """Initialisation function for the data loader

        :param data_dirpath: directory containing the data
        :param img_ids_filepath: file containing the ids of the images to load
        :returns: N/A
        :rtype: N/A

        """
        super().__init__(data_dirpath, img_ids_filepath)
        self.data_dict = defaultdict(dict)

    def load_data(self):
        """ Loads the Adobe5k image data into a Python dictionary

        :returns: Python two-level dictionary containing the images
        :rtype: Dictionary of dictionaries

        """

        logging.info("Loading Adobe5k dataset ...")

        with open(self.img_ids_filepath) as f:
            '''
            Load the image ids into a list data structure
            '''
            image_ids = f.readlines()
            # you may also want to remove whitespace characters like `\n` at the end of each line
            image_ids_list = [x.rstrip() for x in image_ids]

        idx = 0
        idx_tmp = 0
        img_id_to_idx_dict = {}

        for root, dirs, files in os.walk(self.data_dirpath):

            for file in files:

                img_id = file.split("-")[0]

                is_id_in_list = False
                for img_id_test in image_ids_list:
                    if img_id_test == img_id:
                        is_id_in_list = True
                        break

                if is_id_in_list:  # check that the image is a member of the appropriate training/test/validation split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx_tmp = idx
                        idx += 1
                    else:
                        idx_tmp = img_id_to_idx_dict[img_id]

                    if "input" in root:  # change this to the name of your
                                        # input data folder

                        input_img_filepath = file

                        self.data_dict[idx_tmp]['input_img'] = root + \
                            "/" + input_img_filepath

                    elif ("output" in root):  # change this to the name of your
                                             # output data folder

                        output_img_filepath = file

                        self.data_dict[idx_tmp]['output_img'] = root + \
                            "/" + output_img_filepath

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        for idx, imgs in self.data_dict.items():
            assert ('input_img' in imgs)
            assert ('output_img' in imgs)

        return self.data_dict

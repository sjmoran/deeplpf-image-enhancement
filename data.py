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
from abc import abstractmethod


class Dataset(torch.utils.data.Dataset):
    """PyTorch dataset of (input, target) image pairs with lazy loading.

    Wraps a dictionary of image-pair filepaths and loads each pair from disk on
    demand. During training (``is_valid=False``) it applies random horizontal
    and vertical flips as data augmentation; validation, test and inference
    modes load the images unaugmented.
    """

    def __init__(self, data_dict, normaliser=2 ** 8 - 1, is_valid=False, is_inference=False, crop_size=None):
        """Initialisation for the Dataset object

        :param data_dict: dictionary of dictionaries containing images
        :param crop_size: if set, training images are randomly cropped to this
            square size (the same crop is applied to input and target). Needed
            for a batch size greater than one, since FiveK images vary in size
            and the default collation requires a uniform size. Ignored for
            validation and inference.
        :returns: N/A
        :rtype: N/A

        """
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
                if self.normaliser==1:
                    input_img = input_img.astype(np.uint8)
                input_img = TF.to_pil_image(input_img)
                input_img = TF.to_tensor(input_img)

                sample = {'input_img': input_img,
                          'name': self.data_dict[idx]['input_img'].split("/")[-1]}

                # Enhancing your own photographs means there is no retouched
                # target to compare against, so the target is optional here and
                # the caller checks for it. It is not optional for training.
                target_path = self.data_dict[idx].get('output_img')
                if target_path is not None:
                    output_img = util.ImageProcessing.load_image(
                        target_path, normaliser=self.normaliser)
                    if self.normaliser==1:
                        output_img = output_img.astype(np.uint8)
                    output_img = TF.to_pil_image(output_img)
                    sample['output_img'] = TF.to_tensor(output_img)

                return sample

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

    def load_data(self, require_output=True):
        """ Loads the Adobe5k image data into a Python dictionary

        :param require_output: fail if an image has no retouched target. False
                               for inference over images you only have inputs
                               for, where there is nothing to compare against.
        :returns: Python two-level dictionary containing the images
        :rtype: Dictionary of dictionaries

        """

        logging.info("Loading Adobe5k dataset ...")

        with open(self.img_ids_filepath) as f:
            # One image id per line, e.g. "a0001"
            image_ids_list = [x.rstrip() for x in f.readlines()]

        # Image ids are mapped to dense integer indices in the order they are
        # first encountered by os.walk; each index holds an input/output pair.
        idx = 0
        img_id_to_idx_dict = {}

        for root, dirs, files in os.walk(self.data_dirpath):

            for file in files:

                # The id is the filename up to the first "-", e.g.
                # "a0001-jmac_DSC1459.png" -> "a0001". Strip any extension so
                # that a file with no "-" in its name, which is what people's
                # own photographs look like, still matches its listed id
                # instead of silently matching nothing.
                img_id = os.path.splitext(file.split("-")[0])[0]

                if img_id in image_ids_list:  # the image belongs to this train/valid/test split

                    if not img_id in img_id_to_idx_dict.keys():
                        img_id_to_idx_dict[img_id] = idx
                        self.data_dict[idx] = {}
                        self.data_dict[idx]['input_img'] = None
                        self.data_dict[idx]['output_img'] = None
                        idx += 1
                    pair_idx = img_id_to_idx_dict[img_id]

                    # Files are paired by their parent directory name: "input"
                    # for the source image, "output" for the retouched target.
                    if "input" in root:
                        self.data_dict[pair_idx]['input_img'] = root + "/" + file
                    elif ("output" in root):
                        self.data_dict[pair_idx]['output_img'] = root + "/" + file

                else:

                    logging.debug("Excluding file with id: " + str(img_id))

        if not self.data_dict:
            raise FileNotFoundError(
                'none of the %d ids in %s matched an image under %s. An id is '
                'the filename up to the first "-", without its extension, and '
                'the images must sit in an "input" directory (with their '
                'targets, if any, in "output").'
                % (len(image_ids_list), self.img_ids_filepath, self.data_dirpath))

        # These were `assert 'input_img' in imgs`, which a defaultdict entry
        # satisfies while holding None, so a missing file surfaced much later
        # as `'NoneType' object has no attribute 'read'` inside a DataLoader
        # worker. Check the values, and say which image is short of what.
        for idx, imgs in self.data_dict.items():
            name = imgs.get('input_img') or imgs.get('output_img') or ('index %d' % idx)
            if imgs.get('input_img') is None:
                raise FileNotFoundError(
                    'no input image for %s: expected one under an "input" '
                    'directory below %s' % (name, self.data_dirpath))
            if require_output and imgs.get('output_img') is None:
                raise FileNotFoundError(
                    'no target image for %s: expected one under an "output" '
                    'directory below %s. Inference over images you have no '
                    'target for is supported; training and evaluation are not.'
                    % (name, self.data_dirpath))

        return self.data_dict

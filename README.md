# DeepLPF: Deep Local Parametric Filters for Image Enhancement (CVPR 2020)

[Sean Moran](http://www.seanjmoran.com), Pierre Marza, [Steven McDonagh](https://smcdonagh.github.io/), [Sarah Parisot](https://parisots.github.io/), [Greg Slabaugh](http://gregslabaugh.net/)

Huawei Noah's Ark Lab

<p align="center">
<img src="./images/teaser.png" width="80%"/>
</p>
Repository for the paper DeepLPF: Deep Local Parametric Filters for Image Enhancement. Here you will find a link to the code and information on the datasets. Please raise a Github issue if you need assistance of have any questions on the research. 
<p></p>

### [[Paper]](https://arxiv.org/abs/2003.13985) 
### [[Poster]](https://github.com/sjmoran/sjmoran.github.io/blob/490a37c14e5f59db4e1df5034d8cc57444468d1c/pdfs/DeepLPF_CVPR20_poster.pdf) 
### [[Video]](https://www.youtube.com/watch?v=Sxach3FM6FY) 
### [[Supplementary]](http://www.seanjmoran.com/pdfs/DeepLPF_supplementary.pdf) 

<p align="center">
<a href="https://www.youtube.com/watch?v=Sxach3FM6FY" span>
   <img src="./images/youtube-thumbnail.png" width="90%"/>
</a>
<a href="https://github.com/sjmoran/sjmoran.github.io/blob/490a37c14e5f59db4e1df5034d8cc57444468d1c/pdfs/DeepLPF_CVPR20_poster.pdf" span>
   <img src="./images/poster-img.png" width="100%"/>
</a>
</p>

<table>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
      <td><img src="https://github.com/sjmoran/DeepLPF/blob/b614871bb72a6573cd45fc487f5ad2d5e7e6edda/adobe5k_dpe/deeplpf_example_test_input/a4521-kme_0310.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/9404949f046a53b87a04ecc17583d0fa01951176/adobe5k_dpe/deeplpf_example_test_output/a4521-kme_0310.png"/></td> 
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/7b5e10962cead0b0c20e47f7a0c750562f7bfc74/adobe5k_dpe/deeplpf_example_test_inference/a4521-kme_0310_TEST_800_5_PSNR_28.774_SSIM_0.897.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th>
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/76bbc1d5673cf0c90c7316d93fecefa9b5a62052/adobe5k_dpe/deeplpf_example_test_input/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/62d12eff53e30382553f75c32743596a7919deba/adobe5k_dpe/deeplpf_example_test_output/a4869-09-05-19-at-19h05m38s-_MG_9563.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/c75cea3d600136005ea1078db3fc663011c54d8b/adobe5k_dpe/deeplpf_example_test_inference/a4869-09-05-19-at-19h05m38s-_MG_9563_TEST_500_29_PSNR_30.090_SSIM_0.978.jpg"/></td> 
  </tr>
  <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/6202bcdb89f37ade82bbafcf64a46b3d49e68c15/adobe5k_dpe/deeplpf_example_test_input/a4933-Duggan_090428_8040.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/40bceb27866af7a47e3523381af392820bf5a49c/adobe5k_dpe/deeplpf_example_test_output/a4933-Duggan_090428_8040.png"/></td>     <td><img src="https://github.com/sjmoran/DeepLPF/blob/7745ebfb35fc9a6c8e27bbc09c4b82c94a77e632/adobe5k_dpe/deeplpf_example_test_inference/a4933-Duggan_090428_8040_TEST_500_23_PSNR_19.011_SSIM_0.882.jpg"/></td> 
  </tr>
     <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/944e9bfac7ecf7b20b53f1142bf57bf0d2c6bfaf/adobe5k_dpe/deeplpf_example_test_input/a4783-20090321_at_19h16m32__MG_0220.png"/></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/4b5448fc529f9f5f71d117ad6fb54215ae2164c8/adobe5k_dpe/deeplpf_example_test_output/a4783-20090321_at_19h16m32__MG_0220.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/f8b20f0a102c549f716748a3d5e353c48a24d768/adobe5k_dpe/deeplpf_example_test_inference/a4783-20090321_at_19h16m32__MG_0220_TEST_500_2_PSNR_26.362_SSIM_0.905.jpg"/></td> 
  </tr>
        <tr>
    <th>Input</th> 
    <th>Label</th>
    <th>Ours (DeepLPF)</th>
  </tr>
  <tr>
    <td><img src="https://github.com/sjmoran/DeepLPF/blob/fc339bc7177aad138cdd7b3378056b97e6b5afc9/adobe5k_dpe/deeplpf_example_test_input/a4514-kme_0258.png"/></td>   
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/2e2213fbdf07f890e22c40ef2ee31c7cafe98679/adobe5k_dpe/deeplpf_example_test_output/a4514-kme_0258.png"/></td>     
     <td><img src="https://github.com/sjmoran/DeepLPF/blob/2e2213fbdf07f890e22c40ef2ee31c7cafe98679/adobe5k_dpe/deeplpf_example_test_inference/a4514-kme_0258_TEST_500_27_PSNR_27.426_SSIM_0.879.jpg"/></td> 
  </tr>
</table>

### Dependencies

_requirements.txt_ contains the Python packages used by the code.

### How to train DeepLPF and use the model for inference

#### Training DeepLPF

Instructions:

To get this code working on your system / problem you will need to edit the data loading functions, as follows:

1. main.py, change the paths for the data directories to point to your data directory
2. data.py, lines 248, 256, change the folder names of the data input and output directories to point to your folder names

To train, run the command:

```
python3 main.py
```

<p align="center">
<img src="./images/deeplpf_training_loss.png" width="80%"/>
</p>

#### Inference - Using Pre-trained Models for Prediction

The directory _pretrained_models_ contains a set of four DeepLPF pre-trained models on the _Adobe5K_DPE dataset_, each model output from different epochs. The model with the highest validation dataset PSNR (23.94 dB) is at epoch 500:

* deeplpf_validpsnr_23.31_validloss_0.033_testpsnr_23.94_testloss_0.031_epoch_499_model.pt

This model achieves a PSNR of 23.94dB and an SSIM of 0.913 on the Adobe_DPE image dataset. To inference with this model, follow these instructions:

1. Place the images you wish to infer in a directory e.g. ./adobe5k_dpe/deeplpf_example_test_input/. Make sure the directory path has the word "input" somewhere in the path.
2. Place the images you wish to use as groundtruth in a directory e.g. ./adobe5k_dpe/deeplpf_example_test_output/. Make sure the directory path has the word "output" somewhere in the path.
3. Place the names of the images (without extension) in a text file in the directory above the directory containing the images i.e. ./adobe5k_dpe/ e.g. ./adobe5k_dpe/images_inference.txt
4. Run the command and the results will appear in a timestamped directory in the same directory as main.py:

```
python3 main.py --inference_img_dirpath=./adobe5k_dpe/ --checkpoint_filepath=./pretrained_models/deeplpf_validpsnr_23.31_validloss_0.033_testpsnr_23.94_testloss_0.031_epoch_499_model.pt
```

### Bibtex

```
@InProceedings{Moran_2020_CVPR,
author = {Moran, Sean and Marza, Pierre and McDonagh, Steven and Parisot, Sarah and Slabaugh, Gregory},
title = {DeepLPF: Deep Local Parametric Filters for Image Enhancement},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

### Datasets

* __Adobe-DPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). After downloading this dataset you will need to use Lightroom to pre-process the images according to the procedure outlined in the DeepPhotoEnhancer (DPE) [paper](https://github.com/nothinglo/Deep-Photo-Enhancer). Please see the issue [here](https://github.com/nothinglo/Deep-Photo-Enhancer/issues/38#issuecomment-449786636) for instructions. Artist C retouching is used as the groundtruth/target. Note, the images must be extracted in sRGB space. Feel free to raise a Gitlab issue if you need assistance with this (or indeed the Adobe-UPE dataset below). You can also find the training, validation and testing dataset splits for Adobe-DPE in the following [file](https://www.cmlab.csie.ntu.edu.tw/project/Deep-Photo-Enhancer/%5BExperimental_Code_Data%5D_Deep-Photo-Enhancer.zip). The splits can also be found the the [adobe5k_dpe](./adobe5k_dpe/) directory in this repository (note these are a best guess at what the orginal splits from the DPE authors might be).

* __Adobe-UPE__ (5000 images, RGB, RGB pairs): this dataset can be downloaded [here](https://data.csail.mit.edu/graphics/fivek/). As above, you will need to use Lightroom to pre-process the images according to the procedure outlined in the Underexposed Photo Enhancement Using Deep Illumination Estimation (DeepUPE) [paper](https://github.com/wangruixing/DeepUPE) and detailed in the issue [here](https://github.com/wangruixing/DeepUPE/issues/26). Artist C retouching is used as the groundtruth/target. You can find the test images for the Adobe-UPE dataset at this [link](https://drive.google.com/file/d/1HZnNgptNxjKJAhekz2K5yh0mW0yKIws2/view?usp=sharing).

### License

BSD-3-Clause License

### Contributions

We appreciate all contributions. If you are planning to contribute back bug-fixes, please do so without any further discussion.

If you plan to contribute new features, utility functions or extensions to the core, please first open an issue and discuss the feature with us. Sending a PR without discussion might end up resulting in a rejected PR, because we might be taking the core in a different direction than you might be aware of.

# Where these splits come from

`images_train.txt`, `images_valid.txt` and `images_test.txt` are the **original
DPE splits**, recovered 2026-09-08 from a third-party mirror of the Deep Photo
Enhancer release: `MrRobot2211/deep-photo-enhance-t2` at commit `f59c610`, which
predates that repository's TensorFlow-2 port and carries the release verbatim as
`MIT-Adobe/{train_input,train_label,test}.txt`.

Provenance is circumstantial but strong: the surrounding code carries the
original DGX paths (`/tmp3/nothinglo/...`), sets
`FLAGS['process_write_test_img_count'] = 498` matching the DPE README's
`images/MIT498/` output folder, and the file names match those the README
refers to. Every one of the official links on `cmlab.csie.ntu.edu.tw` returns
404, and the Wayback Machine holds only 404 captures, so these lists could not
be obtained from the authors' own distribution.

| | count |
|---|---|
| train (DPE `train_input`) | 2250 |
| valid (DPE `train_label`) | 2250 |
| test  (DPE `test`)        | 498 |

Verified: the three sets are pairwise disjoint, their union is 4998 unique ids,
and the two ids absent from DPE entirely are `a0160` and `a2952`.

Note the test set is **498 images, not 500** - the "500 test images" figure
repeated throughout the literature is off by two.

DPE's supervised variant trains on the 2250 `train_input` ids against their own
Expert-C targets; its `train_label` ids are the second, unpaired half of the GAN
setup. Here they serve as a validation split, which DPE itself does not have -
DPE monitored the test set every half-epoch and selected on running maximum test
PSNR. Using `train_label` for validation is therefore stricter than the original
protocol, not looser.

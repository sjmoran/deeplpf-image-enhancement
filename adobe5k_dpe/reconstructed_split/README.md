# Reconstructed split

`images_train.txt`, `images_valid.txt` and `images_test.txt` here are a
2250 / 2250 / 500 split of the FiveK identifiers, built by taking `a4501`-`a5000`
as the test set. It is **not** the DPE protocol: the DPE test set is 498
scattered identifiers, and the two share 45 images.

The checkpoint in `pretrained_models/` was trained against this split, so its
reported PSNR belongs to this split and cannot be compared with a DPE-protocol
number from the literature. The DPE lists in the directory above are the ones to
use for a comparable result.

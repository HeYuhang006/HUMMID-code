"""Write NIfTI predictions and export derived images."""

import nibabel as nib
import numpy as np


def save_prediction(path, data, reference):
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    nib.save(nib.Nifti1Image(data.astype(np.float32), reference.affine, header), path)


def save_nii(fpNii, data, fpRef):
    new_header = nib.load(fpRef).header.copy()
    new_img = nib.nifti1.Nifti1Image(data, None, header=new_header)
    nib.save(new_img, fpNii)

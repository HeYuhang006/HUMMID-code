# Input for HCP subject `100307`

For the official S1200 download, the preprocessed 3T diffusion measurements, gradients, brain mask and resampled T1w are under `100307/T1w/Diffusion/`. The HCP download is the source for the HUMMID inputs. HUMMID expects prepared derivatives on a common voxel grid, including the selected six diffusion directions, their diffusion tensor eigenvalue maps, the mean DWI, the registered T1w reference and paired diffusion references. Direction selection, tensor fitting and registration take place upstream of this repository's normalization and patch extraction. [HCP S1200 file layout](https://www.humanconnectome.org/storage/app/media/documentation/s1200/HCP_S1200_Release_Appendix_III%2B7T%2BDiffBed.pdf) · [HUMMID input definitions](../../README.md#input-files)

Store the prepared volumes locally as shown below. Use `b = 0` followed by the same six directions in the same order for both 4D DWI files. The NIfTI volumes must share spatial dimensions and affine. The preprocessing module uses the mask to extract 64³ patches from the conformed volume.

```text
example/input/
├── subjects.txt
└── 100307/
    ├── bmask_clean.nii.gz
    ├── 100307_dti_set1_L1.nii.gz
    ├── 100307_dti_set1_L2.nii.gz
    ├── 100307_dti_set1_L3.nii.gz
    ├── 100307_diff_meandwi_corrected.nii.gz
    ├── 100307_dwi6_set1.nii.gz
    ├── 100307_t1w_conform.nii.gz
    ├── 100307_dwi6_ref.nii.gz
    └── 100307_t1w_bmask_conform.nii.gz
```

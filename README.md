# HUMMID-code

**HUMMID: High-fidelity Ultrafast Macrostructural and Microstructural Brain Imaging using Deep Learning**

This repository provides the HCP implementation of the two-stage HUMMID model and its convolutional ablation. The first network synthesizes a T1-weighted volume from six-direction diffusion inputs and derived maps. The second uses the synthesized T1-weighted image to enhance the b = 0 volume and the six diffusion-weighted volumes.

Both HUMMID networks use 3D convolutional block attention modules in the four encoder blocks before pooling. For 64 × 64 × 64 input patches, the encoder attention blocks operate at spatial resolutions of 64³, 32³, 16³ and 8³. Transformer attention operates on the 4³ bottleneck grid of each input patch. Each convolutional decoder upsamples the features and concatenates them with the corresponding encoder skip features before applying a convolutional block. HUMMID and the CNN ablation use LeakyReLU with a negative slope of 0.01 in their convolutional feature blocks. HUMMID also uses LeakyReLU in channel attention and GELU in the transformer MLP.

## Repository layout

| Directory | Role |
| --- | --- |
| [training/](training/) | HUMMID and CNN training entry points, the MAE–SSIM objective, and checkpoint selection |
| [inference/](inference/) | Loads trained weights, runs both stages, and writes NIfTI predictions |
| [dataloader/](dataloader/) | Reads the HCP NIfTI layout, subject lists, and paired patch batches |
| [model/](model/) | Two-stage TransUNet, 3D attention, and the CNN ablation |
| [preprocessing/](preprocessing/) | Brain-mask normalization, patch coordinates, extraction, and reassembly |
| [example/](example/) | One HCP Young Adult subject, input preparation instructions, and a local output directory |

## Setup

Use Python 3.10 or later and install the dependencies in a virtual environment.

```bash
python -m pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation and a GPU are recommended for training. Pass `--device cpu` to select the CPU or `--device cuda:0` to select a GPU.

## Input files

The scripts expect preprocessed, co-registered NIfTI volumes on a common voxel grid. The [HCP example](example/) uses subject `100307` under `example/input/`. For another subject, use its directory name as the filename prefix.

| File | Content |
| --- | --- |
| `bmask_clean.nii.gz` | 3D brain mask for patch placement and intensity normalization |
| `100307_dti_set1_L1.nii.gz`, `100307_dti_set1_L2.nii.gz`, `100307_dti_set1_L3.nii.gz` | Three 3D diffusion-tensor eigenvalue maps from the six-direction input |
| `100307_diff_meandwi_corrected.nii.gz` | 3D mean diffusion-weighted image used by the HCP workflow |
| `100307_dwi6_set1.nii.gz` | 4D diffusion input with seven channels in order, b = 0 followed by the six selected directions |
| `100307_t1w_conform.nii.gz` | Co-registered 3D reference T1-weighted image |
| `100307_dwi6_ref.nii.gz` | 4D reference with seven channels in the same order as the input |
| `100307_t1w_bmask_conform.nii.gz` | 3D T1-weighted brain mask used for the inference output |

The benchmark workflow uses reference intensities to convert normalized predictions back to image intensity units. Prepare the references and masks before running inference. Brain masks must be nonempty and the mask-derived 64 × 64 × 64 patches must fit within the supplied grid.

A split file contains one subject directory name per line, in the order you want to use it. Keep training, validation and test subjects disjoint.

## Preprocessing

The input workflow starts with diffusion volumes, the derived tensor maps, and the brain mask on the same voxel grid. A conformed input can have shape 256 × 256 × 256. The spatial sampling is preserved when forming 64 × 64 × 64 patches.

1. **Normalize the volumes.** For each normalized channel, compute its mean and standard deviation over the brain mask. Subtract the mean, divide by the standard deviation, and mask the result. The mean DWI and each of the seven diffusion input channels are normalized separately. The three tensor eigenvalue channels retain their input values. Training references are normalized independently.
2. **Split into patches.** Use the brain-mask bounding box to place 64³ patches along all three axes. The default adds one patch position per axis to allow overlap. A mask spanning all three axes of a 256³ grid gives 5 × 5 × 5 patch positions with this default. Save the coordinates in memory for reassembly.
3. **Reassemble predictions.** Restore each patch to its source coordinates, crop five voxels from its interior overlapping faces, and average predictions where they overlap. The reference-based HCP workflow rescales intensities using the reference mean and standard deviation, then writes the predictions in the input NIfTI coordinate system.

Both training and inference call these same preprocessing modules. To inspect a single input patch from a prepared example:

```python
import numpy as np

from dataloader import load_subject
from preprocessing import block_indices, extract_block

example = load_subject("example/input", "100307")
coordinates = block_indices(example.brain_mask)
patch = extract_block(example.inputs, coordinates[:1], dtype=np.float32)[0]
print(patch.shape)  # (64, 64, 64, 11)
```

## Train and run inference

```bash
python -m training.trainer --data-root data --train-subjects subject_lists/train.txt --val-subjects subject_lists/val.txt --model-dir checkpoints/hummid
python -m inference.Inference --data-root data --subjects subject_lists/test.txt --model-dir checkpoints/hummid --output-dir runs/hummid
```

Training saves the best validation checkpoints `HCP_WNet_T1w.pth` and `HCP_WNet_DWI.pth`, along with the loss history. Inference writes T1-weighted and diffusion predictions under each subject's output directory in `t1w/` and `dwi/`.

Use the corresponding entry points for the CNN ablation and keep the same subject splits for comparison.

```bash
python -m training.CNN_trainer --data-root data --train-subjects subject_lists/train.txt --val-subjects subject_lists/val.txt --model-dir checkpoints/cnn
python -m inference.CNN_Inference --data-root data --subjects subject_lists/test.txt --model-dir checkpoints/cnn --output-dir runs/cnn
```

Both HCP stages and the CNN ablation minimize `0.8 * MAE + 0.2 * (1 - SSIM)`. Both terms use the brain mask. Global SSIM uses the reference intensity range within each patch and image channel. The seven diffusion channels are weighted equally.

The script defaults are 40 epochs for HUMMID and 40 for the CNN ablation. Use `--epochs`, `--learning-rate`, `--batch-size` and `--no-flip` to configure a run. Train and validation lists are supplied explicitly so the same split can be reused across models.

Supply your own preprocessed inputs and split lists. Use checkpoints trained with the model architectures in this release, or train them using the commands above. The single-subject example demonstrates the reference-based workflow. The source code is distributed under the [MIT License](LICENSE). HCP imaging and derived data are governed by the [HCP Open Access Data Use Terms](https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms).

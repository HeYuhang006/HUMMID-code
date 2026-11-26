"""Run the reference-based HCP inference workflow with local model checkpoints."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataloader import load_subject, read_subjects
from model.factory import checkpoint_paths, load_model_safe, model_pair
from preprocessing import block2brain, block_indices, extract_block, single_layer_denormalize_image

from .io import save_prediction


def predict_subject(t1_model, dwi_model, volumes, device, batch_size=1):
    mask = volumes.brain_mask
    inds = block_indices(mask)
    patches = extract_block(volumes.inputs, inds, dtype=np.float32)
    block_masks = extract_block(mask, inds, dtype=np.float32)
    data = TensorDataset(torch.from_numpy(patches), torch.from_numpy(block_masks))
    t1_blocks, dwi_blocks = [], []
    with torch.inference_mode():
        for inputs, masks in DataLoader(data, batch_size=batch_size, shuffle=False):
            inputs, masks = inputs.to(device), masks.to(device)
            t1w = t1_model(inputs, masks)[..., :-1]
            dwi_features = torch.cat((inputs[..., 4:], t1w), dim=-1)
            dwi = dwi_model(dwi_features, masks)[..., :-1]
            t1_blocks.append(t1w.cpu().numpy())
            dwi_blocks.append(dwi.cpu().numpy())

    t1w_norm, _ = block2brain(np.concatenate(t1_blocks), inds, mask, sz_crop=5)
    dwi_norm, _ = block2brain(np.concatenate(dwi_blocks), inds, mask, sz_crop=5)
    return (
        single_layer_denormalize_image(volumes.reference_t1w, t1w_norm * mask, mask),
        single_layer_denormalize_image(volumes.reference_dwi, dwi_norm * mask, mask),
    )


def main(cnn=False, argv=None):
    parser = argparse.ArgumentParser(description=__doc__ if not cnn else "Run the CNN ablation inference.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True, help="One subject folder name per line.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("batch size must be positive")

    subjects = read_subjects(args.subjects)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this PyTorch environment")
    paths = checkpoint_paths(args.model_dir, cnn)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("missing model checkpoint(s): " + ", ".join(missing))
    t1_model, dwi_model = (
        load_model_safe(model, path, device).eval()
        for model, path in zip(model_pair(cnn), paths, strict=True)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for subject in subjects:
        destination = args.output_dir / subject
        if destination.exists():
            raise FileExistsError(f"Choose a new output folder for {subject}: {destination}")
        volumes = load_subject(args.data_root, subject)
        mask_path = args.data_root / subject / f"{subject}_t1w_bmask_conform.nii.gz"
        t1_mask_image = nib.load(mask_path)
        grid = nib.load(volumes.mask_path)
        if (t1_mask_image.shape != grid.shape
                or not np.allclose(t1_mask_image.affine, grid.affine, atol=1e-3, rtol=1e-5)):
            raise ValueError(f"Expected the same voxel grid as {volumes.mask_path}: {mask_path}")
        t1_mask = t1_mask_image.get_fdata()[..., None]
        if not np.isfinite(t1_mask).all():
            raise ValueError(f"Found non-finite data in {mask_path}")
        t1w, dwi = predict_subject(t1_model, dwi_model, volumes, device, args.batch_size)
        with TemporaryDirectory(dir=args.output_dir, prefix=".hummid-") as staging:
            result = Path(staging) / subject
            t1_dir, dwi_dir = result / "t1w", result / "dwi"
            t1_dir.mkdir(parents=True)
            dwi_dir.mkdir()
            save_prediction(t1_dir / f"{subject}_pred_shared_t1w.nii.gz", t1w * t1_mask, grid)
            save_prediction(dwi_dir / f"{subject}_pred_shared_dwi.nii.gz", dwi, grid)
            result.rename(destination)
        print(f"Saved T1w and DWI predictions for {subject}")


if __name__ == "__main__":
    main()

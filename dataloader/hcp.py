"""I/O and patch preparation for the public HCP reference workflow."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from preprocessing import block_indices, extract_block, normalize_inputs, normalize_targets


def read_subjects(path):
    """Read one subject directory name per line, preserving the supplied order."""
    with Path(path).open(encoding="utf-8-sig") as stream:
        names = [line.strip() for line in stream if line.strip()]
    if not names:
        raise ValueError(f"No subject names in {path}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate subject names in {path}")
    if any(name in {".", ".."} or any(char in name for char in "/\\:") for name in names):
        raise ValueError(f"Expected subject directory names in {path}")
    return names


@dataclass
class SubjectVolumes:
    brain_mask: np.ndarray
    mask_path: Path
    inputs: np.ndarray
    raw_dwi: np.ndarray
    reference_t1w: np.ndarray
    reference_dwi: np.ndarray


def load_subject(data_root, subject):
    """Load the co-registered masked, preprocessed volumes for one subject."""
    folder = Path(data_root) / subject
    mask_path = folder / "bmask_clean.nii.gz"
    mask_image = nib.load(mask_path)
    mask_3d = mask_image.get_fdata()
    if mask_3d.ndim != 3 or not np.isfinite(mask_3d).all() or not (mask_3d > 0.5).any():
        raise ValueError(f"A finite 3D brain mask with foreground is required: {mask_path}")
    mask = mask_3d[..., None]

    def load(name, channels=None):
        path = folder / f"{subject}_{name}.nii.gz"
        image = nib.load(path)
        expected_shape = mask_3d.shape if channels is None else (*mask_3d.shape, channels)
        if image.shape != expected_shape:
            raise ValueError(f"Expected {expected_shape} for {path}, got {image.shape}")
        if not np.allclose(image.affine, mask_image.affine, atol=1e-3, rtol=1e-5):
            raise ValueError(f"Expected the same voxel grid as {mask_path}: {path}")
        data = image.get_fdata()
        if not np.isfinite(data).all():
            raise ValueError(f"Found non-finite data in {path}")
        return data

    anatomy = np.stack([load(f"dti_set1_L{i}") for i in (1, 2, 3)]
                       + [load("diff_meandwi_corrected")], axis=-1)
    raw_dwi = load("dwi6_set1", 7)
    reference_t1w = load("t1w_conform")[..., None]
    reference_dwi = load("dwi6_ref", 7)
    foreground = mask_3d > 0.5
    normalized_values = (anatomy[..., -1:], raw_dwi, reference_t1w, reference_dwi)
    if any(np.any(part[foreground].std(axis=0) == 0) for part in normalized_values):
        raise ValueError(f"Normalization requires nonconstant channels within the brain mask: {subject}")
    inputs = normalize_inputs(anatomy, raw_dwi, mask)
    return SubjectVolumes(mask, mask_path, inputs, raw_dwi, reference_t1w, reference_dwi)


class Nii_dataloader:
    def __init__(self, data_root, train_subjects, valid_subjects, sz_block=64, sz_pad=1, flip=True):
        train_subjects = tuple(train_subjects)
        valid_subjects = tuple(valid_subjects)
        if not train_subjects or not valid_subjects:
            raise ValueError("Training and validation splits must each contain at least one subject")
        if set(train_subjects) & set(valid_subjects):
            raise ValueError("Training and validation splits must be disjoint")
        self.data_root = Path(data_root)
        self.train_subjects = train_subjects
        self.valid_subjects = valid_subjects
        self.sz_block = sz_block
        self.sz_pad = sz_pad
        self.flip = flip

    def _dataset(self, subjects):
        inputs, targets, masks = [], [], []
        for subject in subjects:
            volumes = load_subject(self.data_root, subject)
            mask = volumes.brain_mask
            target = normalize_targets(volumes.reference_t1w, volumes.reference_dwi, mask)
            inds = block_indices(mask, self.sz_block, self.sz_pad)
            for chunks, data in ((inputs, volumes.inputs), (targets, target), (masks, mask)):
                blocks = extract_block(data, inds, dtype=np.float32)
                chunks.append(blocks)
                if self.flip:
                    chunks.append(blocks[:, ::-1, :, :, :])
        return TensorDataset(*(torch.from_numpy(np.concatenate(chunks)) for chunks in (inputs, targets, masks)))

    def create_dataloaders(self, batch_size):
        return (
            DataLoader(self._dataset(self.train_subjects), batch_size=batch_size, shuffle=True),
            DataLoader(self._dataset(self.valid_subjects), batch_size=batch_size, shuffle=False),
        )

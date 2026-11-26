"""Masked intensity normalization and rescaling."""

import copy

import numpy as np


def denormalize_image(imgall, imgnormall, mask):
    imgresall_denorm = np.zeros(imgall.shape)
    ch = imgall.shape[-1]
    mask1 = np.tile(mask, (1, 1, 1, ch))
    img_mean = np.mean(imgall[mask1 > 0.5])
    img_std = np.std(imgall[mask1 > 0.5])
    imgres_norm = (imgnormall * img_std + img_mean) * mask
    imgresall_denorm = imgres_norm
    return imgresall_denorm


def normalize_image(imgall, imgresall, mask, norm_ch='all'):
    imgall_norm = copy.deepcopy(imgall)
    imgresall_norm = copy.deepcopy(imgresall)
    ch = imgall.shape[-1]
    mask1 = np.tile(mask, (1, 1, 1, ch))
    img_mean = np.mean(imgall[mask1 > 0.5])
    img_std = np.std(imgall[mask1 > 0.5])
    img_norm = (imgall - img_mean) / img_std * mask
    imgres_norm = (imgresall - img_mean) / img_std * mask
    imgall_norm = img_norm
    imgresall_norm = imgres_norm
    return imgall_norm, imgresall_norm


def single_layer_denormalize_image(imgall, imgnormall, mask):
    imgresall_denorm = np.zeros(imgall.shape)

    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj: jj + 1]
        imgres = imgnormall[:, :, :, jj: jj + 1]

        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])

        imgres_norm = (imgres * img_std + img_mean) * mask

        imgresall_denorm[:, :, :, jj: jj + 1] = imgres_norm
    return imgresall_denorm


def single_layer_normalize_image(imgall, imgresall, mask, norm_ch='all'):
    imgall_norm = copy.deepcopy(imgall)
    imgresall_norm = copy.deepcopy(imgresall)

    if norm_ch == 'all':
        norm_ch = np.arange(imgall.shape[-1])
    for jj in norm_ch:
        img = imgall[:, :, :, jj: jj + 1]
        imgres = imgresall[:, :, :, jj: jj + 1]

        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])

        img_norm = (img - img_mean) / img_std * mask
        imgres_norm = (imgres - img_mean) / img_std * mask

        imgall_norm[:, :, :, jj: jj + 1] = img_norm
        imgresall_norm[:, :, :, jj: jj + 1] = imgres_norm
    return imgall_norm, imgresall_norm


def add_gauss_noise(image,mask):
    img_mean = np.mean(image[mask > 0.5])
    img_std = np.std(image[mask > 0.5])
    noise = np.random.normal(img_mean, img_std, size=image[:, :, :, 0:6].shape)
    image_noisy = image.copy()
    image_noisy[:, :, :, 0:6] += noise
    return image_noisy


def normalize_inputs(anatomy, raw_dwi, mask):
    """Return [three tensor eigenvalues, mean DWI, b=0, six DWIs]."""
    anatomy_norm, _ = single_layer_normalize_image(anatomy, anatomy, mask, norm_ch=[3])
    dwi_norm, _ = single_layer_normalize_image(raw_dwi, raw_dwi, mask)
    return np.concatenate((anatomy_norm, dwi_norm), axis=-1)


def normalize_targets(reference_t1w, reference_dwi, mask):
    """Return [reference b=0 and six DWIs, reference T1w, brain mask]."""
    ref_dwi, _ = single_layer_normalize_image(reference_dwi, reference_dwi, mask)
    ref_t1w, _ = single_layer_normalize_image(reference_t1w, reference_t1w, mask)
    return np.concatenate((ref_dwi, ref_t1w, mask), axis=-1)

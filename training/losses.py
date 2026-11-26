"""HCP training objective and supporting loss functions."""

import numpy as np
import torch


def mean_squared_error_weighted(y_true, y_pred, loss_weights = 1):
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    return torch.mean((y_pred_weighted - y_true_weighted) ** 2)


def mean_squared_error_weighted_3d(y_true, y_pred, loss_weights = 1):
    # if y_pred.shape == 5 :
    #     y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    #     y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    # if y_true.shape() == 4 :
    y_true_weighted = y_true[:, :, :] * loss_weights
    y_pred_weighted = y_pred[:, :, :] * loss_weights
    return np.mean((y_pred_weighted - y_true_weighted) ** 2)


def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    temp = torch.mean(torch.abs(y_pred_weighted - y_true_weighted))
    return temp


def combined_mae_ssim_loss(prediction, reference, mask, mae_weight=0.8):
    """Combine masked MAE and global structural dissimilarity (1 - SSIM).

    Images have shape (batch, depth, height, width, channels). Compute global
    moments and the reference intensity range for each patch and channel.
    Patches without foreground are excluded. A one-voxel patch contributes MAE.
    """
    if (prediction.ndim != 5 or reference.shape != prediction.shape
            or mask.shape != (*reference.shape[:-1], 1)):
        raise ValueError("Expected matching images and a single-channel brain mask")
    dtype = torch.promote_types(torch.float32, torch.promote_types(prediction.dtype, reference.dtype))
    prediction = prediction.to(dtype=dtype)
    reference = reference.to(dtype=dtype)

    spatial = (1, 2, 3)
    foreground = mask > 0.5
    count = foreground.sum(dim=spatial, keepdim=True)
    normalizer = count.clamp_min(1)
    pred_in = torch.where(foreground, prediction, 0)
    ref_in = torch.where(foreground, reference, 0)
    mae = (pred_in - ref_in).abs().sum(dim=spatial, keepdim=True) / normalizer

    pred_mean = pred_in.sum(dim=spatial, keepdim=True) / normalizer
    ref_mean = ref_in.sum(dim=spatial, keepdim=True) / normalizer
    pred_centered = torch.where(foreground, pred_in - pred_mean, 0)
    ref_centered = torch.where(foreground, ref_in - ref_mean, 0)
    pred_variance = pred_centered.square().sum(dim=spatial, keepdim=True) / normalizer
    ref_variance = ref_centered.square().sum(dim=spatial, keepdim=True) / normalizer
    covariance = (pred_centered * ref_centered).sum(dim=spatial, keepdim=True) / normalizer

    ref_max = reference.masked_fill(~foreground, float("-inf")).amax(dim=spatial, keepdim=True)
    ref_min = reference.masked_fill(~foreground, float("inf")).amin(dim=spatial, keepdim=True)
    data_range = (ref_max - ref_min).clamp_min(1e-6)
    c1, c2 = (0.01 * data_range).square(), (0.03 * data_range).square()
    luminance = (2 * pred_mean * ref_mean + c1) / (pred_mean.square() + ref_mean.square() + c1)
    structure = (2 * covariance + c2) / (pred_variance + ref_variance + c2)
    ssim_loss = torch.where(count >= 2, 1 - luminance * structure, 0)

    patch_loss = mae_weight * mae + (1 - mae_weight) * ssim_loss
    valid_channels = ((count > 0).sum() * prediction.shape[-1]).clamp_min(1)
    return patch_loss.sum() / valid_channels


def Multi_task_loss(y_true, y_pred, t1w_loss_weight = 0.1):
    loss_weights = y_true[:, :, :, :, -1:]
    t1w_pred = y_pred[:, :, :, :, -2:-1] * loss_weights
    dwi6_pred = y_pred[:, :, :, :, :-2] * loss_weights
    t1w_true = y_true[:, :, :, :, -2:-1] * loss_weights
    dwi6_true = y_true[:, :, :, :, :-2] * loss_weights
    t1w_loss = torch.mean(torch.abs(t1w_pred - t1w_true))
    t1w_loss = t1w_loss * t1w_loss_weight
    dwi6_loss = torch.mean(torch.abs(dwi6_pred - dwi6_true))

    return torch.add(t1w_loss, dwi6_loss)


def mean_absolute_error_weighted_3d(y_true, y_pred, loss_weights = 1):
    # loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :] * loss_weights
    y_pred_weighted = y_pred[:, :, :] * loss_weights
    return np.mean(np.abs(y_pred_weighted - y_true_weighted))


def mean_squared_error_weighted_some_layers(y_true, y_pred, loss_weights ,layers):
    # if y_pred.shape == 5 :
    #     y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    #     y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    # if y_true.shape() == 4 :
    y_true_weighted = y_true[:, :, :, layers] * loss_weights
    y_pred_weighted = y_pred[:, :, :, layers] * loss_weights
    return np.mean((y_pred_weighted - y_true_weighted) ** 2)


def mean_squared_error_weighted_Multi_layers(y_true, y_pred, loss_weights ):
    # if y_pred.shape == 5 :
    #     y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    #     y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    # if y_true.shape() == 4 :
    y_true_weighted = y_true[:, :, :, 0:] * loss_weights
    y_pred_weighted = y_pred[:, :, :, 0:] * loss_weights
    return np.mean((y_pred_weighted - y_true_weighted) ** 2)

"""Optional evaluation utilities for reconstructed images."""

import numpy as np
from scipy.ndimage import gaussian_filter


def calculate_angle(vol1, vol2):
    # V1,V2: numpy,shape:(X, Y, Z, 3)
    norm1 = np.sqrt(vol1[:,:,:,0:1]*vol1[:,:,:,0:1]+vol1[:,:,:,1:2]*vol1[:,:,:,1:2]+vol1[:,:,:,2:3]*vol1[:,:,:,2:3])
    vol1 = vol1/(norm1+ 1e-10)
    norm2 = np.sqrt(vol2[:,:,:,0:1]*vol2[:,:,:,0:1]+vol2[:,:,:,1:2]*vol2[:,:,:,1:2]+vol2[:,:,:,2:3]*vol2[:,:,:,2:3])
    vol2 = vol2/(norm2 + 1e-10)
    dot = np.abs(np.sum(vol1 * vol2, axis=-1))
    angle_diff = np.degrees(np.arccos(dot))
    return angle_diff


def calculate_v1_loss(ref_img_path, pred_img_path, mask_path):

    # ref_img = nb.load(ref_img_path)
    # ref_data = ref_img.get_fdata()
    # pred_img = nb.load(pred_img_path)
    # pred_data = pred_img.get_fdata()
    # mask_img = nb.load(mask_path)
    # bmask = mask_img.get_fdata().astype(bool)
    ref_data = ref_img_path
    pred_data = pred_img_path
    bmask = mask_path.astype(bool)
    ang_diff = calculate_angle(ref_data, pred_data)
    ang_diff[~bmask] = 0
    diff_vol = ang_diff[bmask ]
    diff_vol = np.abs(diff_vol)
    return np.nanmean(diff_vol)


def calculate_angle_loss(vol1, vol2, mask=None):

    v1_norm = np.linalg.norm(vol1, axis=-1, keepdims=True)
    v2_norm = np.linalg.norm(vol2, axis=-1, keepdims=True)
    vol1_normalized = vol1 / (v1_norm + 1e-10)  # 避免除以零
    vol2_normalized = vol2 / (v2_norm + 1e-10)

    dot_prod = np.sum(vol1_normalized * vol2_normalized, axis=-1)
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    angle_diff = np.degrees(np.arccos(dot_prod))

    if mask is not None:
        angle_diff = np.where(mask, angle_diff, 0)

    nan_mask = np.isnan(angle_diff)
    inf_mask = np.isinf(angle_diff)
    valid_mask = ~(nan_mask | inf_mask)
    valid_elements = angle_diff[valid_mask]

    return np.nanmean(valid_elements)


def calc_ang(vol1,vol2):
    norm1 = np.sqrt(vol1[:,:,:,0:1]*vol1[:,:,:,0:1]+vol1[:,:,:,1:2]*vol1[:,:,:,1:2]+vol1[:,:,:,2:3]*vol1[:,:,:,2:3])
    vol1 = vol1/norm1
    norm2 = np.sqrt(vol2[:,:,:,0:1]*vol2[:,:,:,0:1]+vol2[:,:,:,1:2]*vol2[:,:,:,1:2]+vol2[:,:,:,2:3]*vol2[:,:,:,2:3])
    vol2 = vol2/norm2
    dotprod = abs(np.sum(vol1*vol2,axis=-1))
    dotprod[dotprod>1] = 1
    return np.arccos(dotprod)/np.pi*180


def calculate_3d_ssim(img1, img2, window_size=11, gaussian_sigma=1.5, data_range=255.0, K1=0.01, K2=0.03, use_gaussian=True):
    """
    计算3D NIfTI数据的SSIM
    Parameters:
        img1, img2 (numpy.ndarray): 输入的3D图像数据
        window_size (int): 高斯窗口的大小
        gaussian_sigma (float): 高斯滤波器的标准差
        data_range (float): 像素值的范围
        K1, K2 (float): 稳定性系数
        use_gaussian (bool): 是否使用高斯加权
    Returns:
        ssim_score (float): 全局SSIM值
        ssim_map (numpy.ndarray): 局部SSIM图
    """
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if use_gaussian:
        x = np.linspace(-window_size//2 + 1, window_size//2, window_size)
        g = np.exp(-(x**2) / (2*gaussian_sigma**2))
        window = np.outer(g, g)
        window = np.outer(window, g)
        window /= np.sum(window)
    else:
        window = np.ones((window_size, window_size, window_size)) / window_size**3

    mu1 = gaussian_filter(img1, sigma=gaussian_sigma, mode='reflect')
    mu2 = gaussian_filter(img2, sigma=gaussian_sigma, mode='reflect')

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1**2, sigma=gaussian_sigma, mode='reflect') - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma=gaussian_sigma, mode='reflect') - mu2_sq
    sigma12 = gaussian_filter(img1*img2, sigma=gaussian_sigma, mode='reflect') - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator
    ssim_score = np.mean(ssim_map)

    return ssim_score, ssim_map

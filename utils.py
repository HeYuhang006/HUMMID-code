"""Compatibility imports for the original utility API."""

import os

from inference.io import (
    save_nii,
)
from inference.metrics import (
    calculate_angle,
    calculate_v1_loss,
    calculate_angle_loss,
    calc_ang,
    calculate_3d_ssim,
)
from model.factory import (
    load_model_safe,
)
from preprocessing.normalization import (
    denormalize_image,
    normalize_image,
    single_layer_denormalize_image,
    single_layer_normalize_image,
    add_gauss_noise,
)
from preprocessing.patches import (
    block_ind,
    extract_block,
    block2brain,
)
from training.losses import (
    mean_squared_error_weighted,
    mean_squared_error_weighted_3d,
    mean_absolute_error_weighted,
    combined_mae_ssim_loss,
    Multi_task_loss,
    mean_absolute_error_weighted_3d,
    mean_squared_error_weighted_some_layers,
    mean_squared_error_weighted_Multi_layers,
)


def rootpath():
    current_depth = 1  # 获取当前路径的上级路径
    curr_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本的路径
    fsep = os.sep  # 获取路径分隔符
    parts = curr_path.split(fsep)  # 将路径按分隔符分割成列表
    p = fsep.join(parts[:-current_depth])  # 根据 current_depth 获取上级路径
    if p.endswith(fsep):  # 如果路径以分隔符结尾，则去掉最后一个字符
        p = p[:-1]
    return p

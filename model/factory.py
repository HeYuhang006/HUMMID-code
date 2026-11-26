"""Build the two-stage networks and load compatible checkpoints."""

from pathlib import Path

import torch

from .CNN_model import PureConvUNet
from .TransUnet import TransUNet


def model_pair(cnn=False):
    model_class = PureConvUNet if cnn else TransUNet
    kwargs = {"base_channels": 64}
    if not cnn:
        kwargs.update(num_heads=8, depth=3)
    return (
        model_class(in_channels=11, out_channels=1, **kwargs),
        model_class(in_channels=8, out_channels=7, **kwargs),
    )


def checkpoint_paths(model_dir, cnn=False):
    model_dir = Path(model_dir)
    prefix = "HCP_CNN" if cnn else "HCP_WNet"
    return model_dir / f"{prefix}_T1w.pth", model_dir / f"{prefix}_DWI.pth"


def load_model_safe(model, path, device):
    # 先加载到CPU
    state_dict = torch.load(path, map_location='cpu')
    
    # 处理DataParallel前缀
    new_state_dict = {}
    for key, value in state_dict.items():
        # 移除'module.'前缀
        if key.startswith('module.'):
            key = key[7:]
        new_state_dict[key] = value
    
    # 加载状态字典
    model.load_state_dict(new_state_dict)
    return model.to(device)

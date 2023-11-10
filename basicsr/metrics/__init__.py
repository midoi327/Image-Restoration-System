# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim, calculate_ssim_left, calculate_psnr_left, calculate_skimage_ssim, calculate_skimage_ssim_left

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_ssim_left', 'calculate_psnr_left', 'calculate_skimage_ssim', 'calculate_skimage_ssim_left']


############## HAT 모델 추가 #####################
from copy import deepcopy
from basicsr.utils.registry import METRIC_REGISTRY
def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
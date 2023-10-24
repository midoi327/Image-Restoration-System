# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import numpy as np
# from basicsr.data import create_dataloader, create_dataset
import basicsr.models
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str


import os
import glob
import cv2
from PIL import Image

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    
    
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    # img_path = opt['img_path'].get('input_img')
    # output_path = opt['img_path'].get('output_img')
    
    input_path = os.path.join('demo', 'Set12') # Set12 폴더를 input으로 사용 
    output_path = os.path.join('demo', 'Output') # Output 폴더에 denoising 이미지 저장 

    files_source = glob.glob(os.path.join('demo','Set12', '*.png')) # 테스트하려는 이미지 
    files_source.sort()
    
    for f in files_source:
        
        ## 1. read image
        file_client = FileClient('disk')
        img_bytes = file_client.get(f, None)
        
        try:
            img = imfrombytes(img_bytes, float32=True)
            
        except:
            raise Exception("path {} not working".format(f))

        img = img2tensor(img, bgr2rgb=True, float32=True)
        
        
        ## 2. run inference
        opt['dist'] = False
        model = create_model(opt)
        
        model.feed_data(data={'lq': img.unsqueeze(dim=0)})
        
        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()

        if model.opt['val'].get('grids', False):
            model.grids_inverse()
            
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])
        # sr_img = Image.fromarray(sr_img)
        
        output_filename = os.path.join(output_path, os.path.basename(f))
        cv2.imwrite(output_filename, sr_img)
        

    # print(f'inference {input_path} .. finished. saved to {output_filename}')

if __name__ == '__main__':
    main()


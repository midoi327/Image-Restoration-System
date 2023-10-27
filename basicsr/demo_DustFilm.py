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
from niqe import *


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def normalize(data):
    return data/255.

def resize_image(input_image_path, target_size):
    # 이미지 열기
    image = Image.open(input_image_path)
    # 이미지 리사이징
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    return resized_image


def main():
    
    
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    # img_path = opt['img_path'].get('input_img')
    # output_path = opt['img_path'].get('output_img')
    
    input_path = os.path.join('demo', 'Dust') # Set12 폴더를 input으로 사용 
    output_path = os.path.join('demo', 'Output_Dust') # Output 폴더에 denoising 이미지 저장 

    files_source = glob.glob(os.path.join('demo','Dust', '*.tif')) # 테스트하려는 이미지 
    files_source.sort()
    
    # (Dust 데이터셋을 위한) 이미지 리사이즈할 목표 크기 .. 다른 데이터셋에는 필요없음
    target_size = (200, 600)
    
    niqe_test_before = 0 # 노이즈 제거 전 데이터셋 NIQE 평균 점수
    niqe_test_after = 0 # 노이즈 제거 후 데이터셋 NIQE 평균 점수
    
    
    for f in files_source:
        
        ## 1. read image
        # file_client = FileClient('disk')
        # img_bytes = file_client.get(f, None) # bytes 타입
        
        try:
            # img = imfrombytes(img_bytes, float32=True) # numpy.ndarray 타입
            img = cv2.imread(f)
            img = resize_image(f, target_size)
            img = np.array(img) # shape 가 (600, 200, 3) 이고 값이 0~255 인 이미지 array
            
            # 노이즈 이미지의 경우: (600, 200) 흑백 이미지를 (600, 200, 3) 컬러 이미지로 변환
            if len(img.shape) == 2:
                img = np.stack( (img,)*3, axis= -1)
            
            input_filename = os.path.join(os.path.join('demo', 'Dust_before'), os.path.basename(f))
            cv2.imwrite(input_filename, img)

        except:
            raise Exception("path {} not working".format(f))

        img = img2tensor(img, bgr2rgb=True, float32=True)
        img = normalize(img[:,:,:]) # shape가 [600, 200, 3] 이고 값이 0~1인 이미지 tensor로 변환
        
        ## 2. run inference
        opt['dist'] = False
        model = create_model(opt)
        
        model.feed_data(data={'lq': img.unsqueeze(dim=0)}) # (1, 600, 200, 3)
        
        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()
        
        if model.opt['val'].get('grids', False):
            model.grids_inverse()

        
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])  # cpu().numpy() 들어있음 shape: (600, 200, 3)
        
        output_filename = os.path.join(output_path, os.path.basename(f))
        cv2.imwrite(output_filename, sr_img, params=None)
        print(f'saved to {output_filename}')
        
        
        ##3. niqe 점수 계산하기
        
        # rgb 이미지를 그레이스케일로 변환
        
        img_np = img.permute(1,2,0)# torch.Size([481, 321, 3])
        gray_img= (img_np.cpu().numpy()[:,:,0]*255).astype(np.uint8) # (600, 200)
        gray_sr_img = (sr_img[:,:,0]*255).astype(np.uint8) # (600, 200)
        
        niqe_score_before = niqe(gray_img)
        niqe_score_after = niqe(gray_sr_img)
        
        niqe_test_before += niqe_score_before
        niqe_test_after += niqe_score_after
        
        print(f'전 NIQE: {niqe_score_before: .3f}')
        print(f'후 NIQE: {niqe_score_after: .3f}')
        
    print('sum:', niqe_test_before, niqe_test_after)
        
    niqe_test_before /= len(files_source)
    niqe_test_after /= len(files_source)
    print('\n평균 노이즈 제거 전 NIQE 점수 %.3f' %niqe_test_before)
    print('평균 노이즈 제거 후 NIQE 점수 %.3f' %niqe_test_after)
        
    

if __name__ == '__main__':
    main()


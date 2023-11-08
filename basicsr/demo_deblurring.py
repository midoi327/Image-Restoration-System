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


def normalize(data):
    return data/255.

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def main():
    
    
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()
    
    input_path = os.path.join('demo', 'Deblurring_in') # Set12 폴더를 input으로 사용 
    output_path = os.path.join('demo', 'Deblurring_out') # Output 폴더에 denoising 이미지 저장 

    files_source = glob.glob(os.path.join(input_path, '*')) # 테스트하려는 이미지 
    files_source.sort()
    
    niqe_test_before = 0 # 노이즈 제거 전 데이터셋 NIQE 평균 점수
    niqe_test_after = 0 # 노이즈 제거 후 데이터셋 NIQE 평균 점수
    min_niqe_score = 100 # NIQE 최소 점수 값
    
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
        
        model.feed_data(data={'lq': img.unsqueeze(dim=0)}) # (1, 481, 321, 3)
        
        if model.opt['val'].get('grids', False):
            model.grids()

        model.test()
        
        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])  # cpu().numpy() 들어있음 
        # print('sr_img 정보:', sr_img.shape) # img.shape: (481, 321, 3)
        # sr_img = Image.fromarray(sr_img)
        
        output_filename = os.path.join(output_path, os.path.basename(f))
        cv2.imwrite(output_filename, sr_img)
        print(f'saved to {output_filename}')
        
        
        ##3. niqe 점수 계산하기
        
        # rgb 이미지를 그레이스케일로 변환
        
        img_np = img.permute(1,2,0)# torch.Size([481, 321, 3])
        # gray_img= (img_np.cpu().numpy()[:,:,0]*255).astype(np.uint8) # (481, 321)
        gray_img= (img_np.cpu().numpy()[:,:,:]*255).astype(np.uint8)
        gray_sr_img = (sr_img[:,:,:]*255).astype(np.uint8) # (481, 321)
        
        
        height, width, _ = gray_sr_img.shape # 만약 192x192 보다 크기가 작다면, 해당 값을 200으로 수정
        # 이미지 크기가 193 이하인 경우 조정
        if width <= 193 or height <= 193:
            print(f'{f}이미지의 크기를 조정합니다.')
            width = max(width, 200)
            height = max(height, 200)
            
        gray_sr_img = cv2.resize(gray_sr_img, (width, height))
        
        if gray_sr_img.shape[0] <= 192 or gray_sr_img.shape[1] <= 192:
            print(f'{f}의 크기가 192보다 작습니다. 사이즈 {gray_sr_img.shape[0]}x{gray_sr_img.shape[1]}')
            
        
        niqe_score_before = niqe(gray_img)
        niqe_score_after = niqe(gray_sr_img)
        
        niqe_test_before += niqe_score_before
        niqe_test_after += niqe_score_after
        
        print(f'전 NIQE: {niqe_score_before: .3f}')
        print(f'후 NIQE: {niqe_score_after: .3f}')
        
        if niqe_score_after <= min_niqe_score:
            print(f'{f}의 NIQE 점수는 {niqe_score_after}로 최저 점수를 갱신하였습니다.')
            min_filename = f
            min_niqe_score = niqe_score_after
            

        
    niqe_test_before /= len(files_source)
    niqe_test_after /= len(files_source)
    print('\n평균 노이즈 제거 전 NIQE 점수 %.3f' %niqe_test_before)
    print('평균 노이즈 제거 후 NIQE 점수 %.3f' %niqe_test_after)
    print(f'최소 NIQE 점수: {min_filename}이미지의 {min_niqe_score}점')
        
    

if __name__ == '__main__':
    main()


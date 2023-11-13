# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import numpy as np
import basicsr.models
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite
import os
import glob
import cv2
from PIL import Image
from niqe import *


from os import path as osp
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs
import logging
from basicsr.utils.options import dict2str
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import build_model


from maxim import run_maxim

# (터미널에 순서대로)
# $export PYTHONPATH=/home/piai/문서/miryeong/Multi/:/home/piai/문서/miryeong/Multi/basicsr
# basicsr 모델 모듈 설치 $python setup_basicsr.py develop --no_cuda_ext
# maxim 모델 모듈 설치 $python setup_maxim.py devleop 
# $python basicsr/demo_Multi.py



def normalize(data):
    return data/255.

def nafnet(output_path, files_source, opt):
    
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
        
        output_filename = os.path.join(output_path, os.path.basename(f))
        cv2.imwrite(output_filename, sr_img)
        print(f'{f}가 저장되었습니다.')
        
        
        ##3. niqe 점수 계산하기
        
        # rgb 이미지를 그레이스케일로 변환
        
        img_np = img.permute(1,2,0)# torch.Size([481, 321, 3])
        gray_img= (img_np.cpu().numpy()[:,:,:]*255).astype(np.uint8)
        gray_sr_img = (sr_img[:,:,:]*255).astype(np.uint8) # (481, 321)
        
        gray_img = np.array(Image.fromarray(cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)).convert('L')) # 회색 이미지로 변경
        gray_sr_img = np.array(Image.fromarray(cv2.cvtColor(gray_sr_img, cv2.COLOR_BGR2RGB)).convert('L'))
        
        
        height, width = gray_sr_img.shape # 만약 192x192 보다 크기가 작다면, 해당 값을 200으로 수정
        # 이미지 크기가 193 이하인 경우 조정
        if width <= 193 or height <= 193:
            # print(f'{f}이미지의 크기를 조정합니다.')
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
            # print(f'{f}의 NIQE 점수는 {niqe_score_after}로 최저 점수를 갱신하였습니다.')
            min_filename = f
            min_niqe_score = niqe_score_after
            
    niqe_test_before /= len(files_source)
    niqe_test_after /= len(files_source)
    
    print(f'최소 NIQE 점수: {min_filename}이미지의 {min_niqe_score}점')

    return niqe_test_before, niqe_test_after

def hat(opt):
    torch.backends.cudnn.benchmark = True
    # mkdir and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    # logger.info(get_env_info())
    # logger.info(dict2str(opt))

    # create test dataset and dataloader
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(
            test_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=None, seed=opt['manual_seed'])
        logger.info(f"Number of test images in {dataset_opt['name']}: {len(test_set)}")
        test_loaders.append(test_loader)
    
    model = build_model(opt)
    
    for test_loader in test_loaders:
        test_set_name = test_loader.dataset.opt['name']
        logger.info(f'Testing {test_set_name}...')
        model.validation(test_loader, current_iter=opt['name'], tb_logger=None, save_img=opt['val']['save_img'])

    
    return 1

def maxim():
    run_maxim()

def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    
    input_path = os.path.join('demo', 'Multi_in') # Input path: Multi_in 폴더
    output_path = os.path.join('demo', 'Multi_out') # Output path: Multi_out 폴더

    files_source = glob.glob(os.path.join(input_path, '*')) # 테스트하려는 이미지 
    files_source.sort()
    
    
    # 원하는 옵션 선택 1:denoising 2:deblurring 3:super resolution
    mode = int(input('어떤 작업을 실행하시겠습니까? 원하는 옵션을 입력하세요. 1:denosing 2:deblurring 3:super-resolution 4:dehazing\n'))
    if mode == 1:
        opt = 'options/test/SIDD/NAFNet-width32.yml' # denoising
        opt = parse_options(opt, is_train=False)
        # parse options, set distributed setting, set ramdom seed
        opt['num_gpu'] = torch.cuda.device_count()
        niqe_before, niqe_after = nafnet(output_path, files_source, opt)
        
        # print(f'평균 NIQE 점수는 {niqe_before:.3f}점에서 {niqe_after:.3f}점으로 갱신되었습니다.')
    
    elif mode == 2:
        opt = 'options/test/REDS/NAFNet-width64.yml' # deblurring
        opt = parse_options(opt, is_train=False)
        opt['num_gpu'] = torch.cuda.device_count()
        niqe_before, niqe_after = nafnet(output_path, files_source, opt)
        print('deblurring 작업이 완료되었습니다.')
        
        # print(f'평균 NIQE 점수는 {niqe_before:.3f}점에서 {niqe_after:.3f}점으로 갱신되었습니다.')
    

    elif mode == 3:
        opt = 'options/test/HAT/HAT_SRx4_ImageNet-LR.yml' # super resolution
        opt = parse_options(opt, is_train=False)
        hat(opt)
        print('super resolution 작업이 완료되었습니다.')
        
    
    elif mode == 4:
        maxim() # dehazing
        print('dehazing 작업이 완료되었습니다.')
        
        
if __name__ == '__main__':
    main()


# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch

# from basicsr.data import create_dataloader, create_dataset
import basicsr.models
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding, tensor2img, imwrite

# from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
#                            make_exp_dirs)
# from basicsr.utils.options import dict2str

def main():
    
    
    # parse options, set distributed setting, set ramdom seed
    opt = parse_options(is_train=False)
    opt['num_gpu'] = torch.cuda.device_count()

    img_path = opt['img_path'].get('input_img')
    output_path = opt['img_path'].get('output_img')
    
    ## 1. read image
    file_client = FileClient('disk')

    img_bytes = file_client.get(img_path, None)
    print('img_bytes 정보:', type(img_bytes))
    try:
        img = imfrombytes(img_bytes, float32=True)
        print('imfrombytes 정보:', type(img_bytes))
    except:
        raise Exception("path {} not working".format(img_path))

    img = img2tensor(img, bgr2rgb=True, float32=True)
    print('img 정보:', img)


    ## 2. run inference
    opt['dist'] = False
    model = create_model(opt)

    model.feed_data(data={'lq': img.unsqueeze(dim=0)})

    if model.opt['val'].get('grids', False):
        model.grids()
        print('여기지나감')

    model.test()
    print('모델 테스트 끝남')

    if model.opt['val'].get('grids', False):
        model.grids_inverse()
        print('여기도지나감')

    visuals = model.get_current_visuals() 
    print('///////////////visuals result:', [visuals['result']])
    sr_img = tensor2img([visuals['result']]) # tensor to list to image
    imwrite(sr_img, output_path)

    print(f'inference {img_path} .. finished. saved to {output_path}')

if __name__ == '__main__':
    main()


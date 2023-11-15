# Image Restoration System
**🚀 POSTECH Institute of Artificial Intelligence Intern Program**

-- 이미지 개선 4가지 작업을 딥러닝 모델을 이용하여 파이썬 코드로 구현 (NAFNet, HAT, MAXIM 모델) --

* **Image denosing** with NAFNet https://github.com/megvii-research/NAFNet
* **Image deblurring** with NAFNet 
* **Image super resolution** with HAT https://github.com/XPixelGroup/HAT.git
* **Image dehazing** with MAXIM https://github.com/google-research/maxim.git

* **NAFNet 논문 리뷰 및 요약**

https://drive.google.com/file/d/1AoUg8Lne4XmBxQbV37W_F4qDiyBUiV2S/view?usp=sharing


---

## 📌 **시스템 구조도**
![1231241 (1)](https://github.com/midoi327/Image-Restoration-System/assets/50612011/6c351445-94dd-4986-93a2-e1bb8f871a3f)



---


## 📌 **Required**
**demo_Multi.py 실행시키기 위한 준비 과정**
1. **NAFNet pretrained model** 다운로드 ```NAFNet-width32.yml```
2. **HAT pretrained model** 다운로드 ```HAT_SRx4_ImageNet-pretrain.pth```
3. **MAXIM pretrained model** 다운로드 ```Dehazing-RESIDE-Outdoor```
4. 다운로드 후 experiments에 모델 파일 저장
5. python path 설정 : ```export PYTHONPATH= /프로젝트디렉토리/:/basicsr모듈디렉토리/```
6. ```echo $PYTHONPATH``` : 파이썬 모듈 찾는 경로가 잘 설정되었는지 확인
7. ```python setup_basicsr.py develop --no_cuda_ext``` : processing dependencies for basicsr 모듈
8. ```python setup_maxim.py develop``` : processing dependencies for maxim 모듈


---

## **📌 Quick Start**
**denoising, deblurring 옵션을 원하는 대로 선택하는 demo.Multi.py 사용 방법**

1. Required 실행 조건 만족
2. demo/Multi_in 폴더에 노이즈 이미지 넣어놓기
3. 
```ruby 
python basicsr/demo_Multi.py
``` 
4. ***1: denoising , 2: deblurring, 3: super-resolution 4:dehazing***  옵션 중 원하는 옵션 입력
5. demo/Multi_out 폴더에 작업 후 이미지 생성된다.

---

## **🖇️ FID300 데이터셋 이미지 테스트 수행 결과**








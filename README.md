# NAFNet_Image-denoising
POSTECH Institute of Artificial Intelligence Intern Program

Image denosing with NAFNet https://github.com/megvii-research/NAFNet

![image](https://github.com/midoi327/NAFNet_Image-denoising/assets/50612011/0018df3a-69c0-41b9-a17f-10af795e5ead)


---


**Required:**
**demo.py 실행시키기 위한 준비 과정**
1. pretrained models 다운로드 https://github.com/megvii-research/NAFNet/#results-and-pre-trained-models:~:text=the%20predicted%20image-,pretrained%20models,-should%20be%20downloaded
2. 다운로드 후 experiments에 모델 파일 저장
3. python path 설정 : export PYTHONPATH= /프로젝트디렉토리/:/basicsr모듈디렉토리/
4. echo $PYTHONPATH : 파이썬 모듈 찾는 경로가 잘 설정되었는지 확인
5. python setup.py develop --no_cuda_ext : processing dependencies for basicsr 모듈


---

**🏆 평가 지표**


NIQE | Naturalness Image Quality Evaluator

영상의 자연스러움 및 품질을 평가한다. 

입력 영상의 통계적 특성과 밝기, 색상, 대비 등과 관련된 특징을 추출한다. 사전 학습된 모델을 사용하여 특징을 추출하고 점수를 반환한다. 
점수가 낮을 수록 영상의 품질이 높다. 

https://github.com/guptapraful/niqe

---


**☑️ 노이즈가 추가된 이미지와 원본 이미지에 대해 각각 테스트 수행한 NIQE 점수 결과**

<img width="723" alt="image" src="https://github.com/midoi327/NAFNet_Image-denoising/assets/50612011/34cdd4ae-215a-43a1-a83b-051bb2696e61">


* NAFNet-width32 모델은 Set12, BSD68 노이즈 데이터셋에 대하여 평균 10.94%, 16.40% 의 이미지 개선 성능을 보였다.
* NAFNet-width32 모델은 FID300, Dust_Film 데이터셋에 대하여 이미지 개선 성능을 나타내지 않았다.
* NAFNet-width32 모델은 특히 FID300, Dust_Film 노이즈 데이터셋에 대하여 육안으로 이미지 손상을 확인할 수 있었다.


---

**🖇️ 노이즈가 추가된 이미지와 원본 이미지에 대해 각각 테스트 수행 결과**

![image](https://github.com/midoi327/NAFNet_Image-denoising/assets/50612011/b8817c9f-820f-4915-8844-a89a31f6903f)

![image](https://github.com/midoi327/NAFNet_Image-denoising/assets/50612011/0c9e45a1-1acb-4810-8198-af3979937583)











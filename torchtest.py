import torch

print('Pytorch 버전:', torch.__version__)

# GPU 사용 가능 여부 확인
if torch.cuda.is_available():
    device = torch.device("cuda")  # GPU를 사용할 수 있는 경우 GPU를 사용
    print("GPU 사용 가능")
else:
    device = torch.device("cpu")   # GPU를 사용할 수 없는 경우 CPU를 사용
    print("GPU 사용 불가능")

print("사용하는 장치:", device)




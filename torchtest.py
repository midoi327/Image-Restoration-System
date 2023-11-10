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


import GPUtil

def get_gpu_info():
    gpu_list = GPUtil.getGPUs()
    for gpu in gpu_list:
        print(f"GPU ID: {gpu.id}, GPU Name: {gpu.name}")
        print(f"GPU Memory Free: {gpu.memoryFree} MB")
        print(f"GPU Memory Used: {gpu.memoryUsed} MB")
        print(f"GPU Memory Total: {gpu.memoryTotal} MB")

get_gpu_info()

print(torch.backends.cudnn.version()) # 8700 버전
print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())
torch.cuda.empty_cache()

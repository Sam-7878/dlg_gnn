import torch
if torch.cuda.is_available():
    print("CUDA:", torch.cuda.get_device_name(0))
else:
    print("CPU only")

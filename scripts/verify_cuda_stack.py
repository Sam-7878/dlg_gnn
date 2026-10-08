#!/usr/bin/env python3
import torch
import torch_geometric
import pygod

print("=" * 60)
print("PYTHON & ACCELERATOR STACK QUALIFICATION")
print("=" * 60)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Device Count: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
    cap = torch.cuda.get_device_capability(i)
    print(f"  [GPU {i}] {name} | {mem_gb:.2f} GB | Compute Capability: {cap[0]}.{cap[1]}")

print("-" * 60)
print(f"PyTorch Geometric (PyG) Version: {torch_geometric.__version__}")
print(f"PyGOD Version: {pygod.__version__}")
print("=" * 60)

# Quick tensor check on GPU 0 and GPU 1
for i in range(torch.cuda.device_count()):
    x = torch.randn(100, 100, device=f"cuda:{i}")
    y = torch.matmul(x, x)
    assert y.is_cuda and not torch.isnan(y).any()
    print(f"GPU {i} Matmul Test: PASSED")
print("ALL DEVICE TENSOR TESTS PASSED")

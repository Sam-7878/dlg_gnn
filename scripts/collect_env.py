import sys, platform, subprocess
sys.path.insert(0, "src")
sys.path.insert(0, ".")

try:
    import torch
    torch_ver = torch.__version__
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_available else "N/A"
except ImportError:
    torch_ver = "NOT INSTALLED"
    cuda_available = False
    cuda_ver = "N/A"

try:
    import psutil
    mem = psutil.virtual_memory()
    total_gb = round(mem.total / 1024**3, 1)
    avail_gb = round(mem.available / 1024**3, 1)
except ImportError:
    total_gb = "N/A"
    avail_gb = "N/A"

try:
    cpu_info = subprocess.check_output(["lscpu"], text=True)
    cpu_model = [l for l in cpu_info.split("\n") if "Model name" in l]
    cpu_model = cpu_model[0].split(":")[1].strip() if cpu_model else platform.processor()
except Exception:
    cpu_model = platform.processor()

print(f"torch_version: {torch_ver}")
print(f"cuda_available: {cuda_available}")
print(f"cuda_version: {cuda_ver}")
print(f"python_version: {sys.version.split()[0]}")
print(f"platform: {platform.platform()}")
print(f"cpu_model: {cpu_model}")
print(f"ram_total_gb: {total_gb}")
print(f"ram_avail_gb: {avail_gb}")

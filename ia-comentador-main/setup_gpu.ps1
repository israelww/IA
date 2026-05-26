Write-Host "Instalando PyTorch con CUDA (cu128)..." -ForegroundColor Cyan
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch

Write-Host "Verificando CUDA en PyTorch..." -ForegroundColor Cyan
@'
import torch
print("torch_version:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("cuda_device_name:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = torch.randn(1024, 1024, device="cuda")
    print("gpu_test_mean:", (x @ y).mean().item())
'@ | python -

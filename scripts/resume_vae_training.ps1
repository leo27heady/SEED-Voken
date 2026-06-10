# Resume SQ-VAE-2 v2 training from a checkpoint (64px Shapes3D, T=13).
# Usage (from repo root):
#   .\venv\Scripts\Activate.ps1
#   $env:PYTHONPATH = (Get-Location).Path
#   .\scripts\resume_vae_training.ps1 -Ckpt "C:\path\to\epoch=67-step=56712.ckpt"

param(
    [string]$Config = "configs/Open-MAGVIT2/gpu/shapes3d_sqvae2_64_S_v2.yaml",
    [Parameter(Mandatory = $true)]
    [string]$Ckpt
)

if (-not (Test-Path $Ckpt)) {
    Write-Error "Checkpoint not found: $Ckpt"
    exit 1
}

python main.py fit --config $Config --ckpt_path $Ckpt

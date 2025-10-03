# Quick Start Guide - Windows

## First Time Setup

1. **Ensure you have installed:**
   - Python 3.10 or 3.11
   - NVIDIA GPU drivers (latest)
   - CUDA Toolkit 12.x
   - Visual Studio Build Tools 2022

2. **Create and activate virtual environment:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```powershell
   pip install --upgrade pip
   pip install numpy pycuda opencv-python pygame
   ```

## Running CUDA Programs

### Why Are Helper Scripts Needed?

On Windows, CUDA's compiler (`nvcc`) needs Visual Studio's C++ compiler (`cl.exe`), but it's not in your PATH by default. The two PowerShell scripts solve this problem in different ways:

- **`setup_cuda_env.ps1`**: Sets up the environment **once per session**. Run it when you open PowerShell, then run multiple scripts.
- **`run_with_msvc.ps1`**: Sets up the environment **per script**. No setup needed, but slightly slower for repeated runs.

### Method 1: Setup Environment Once (Recommended)

**Best for:** Running multiple scripts, active development

```powershell
# Run this once per PowerShell session (note the dot-space at start)
. .\setup_cuda_env.ps1

# Then run any script normally
python hello_cuda.py
python minimal_cuda.py
python simplest_cuda_demo.py
python app.py
```

**How it works:** Imports Visual Studio compiler paths into your current PowerShell session. Environment persists until you close the window.

### Method 2: Use Helper Script (No Setup)

**Best for:** Quick single runs, beginners, sharing with others

```powershell
# Automatically sets up environment for each run
.\run_with_msvc.ps1 hello_cuda.py
.\run_with_msvc.ps1 app.py
```

**How it works:** Creates a temporary environment for each script execution. No setup needed, environment doesn't persist.

### Method 3: Developer Command Prompt

**Best for:** Advanced users, alternative to helper scripts

1. Open "Developer Command Prompt for VS 2022" from Start Menu
2. Navigate to project:
   ```cmd
   cd C:\Users\Owner\source\repos\LiteObject\CUDA-Image-Processing-App
   ```
3. Activate virtual environment:
   ```cmd
   .venv\Scripts\activate.bat
   ```
4. Run scripts:
   ```cmd
   python hello_cuda.py
   ```

**How it works:** Microsoft provides pre-configured command prompts with Visual Studio environment already set up.

---

**💡 Quick Recommendation:**
- **First time?** Use Method 2 (`run_with_msvc.ps1`) - simplest
- **Running multiple scripts?** Use Method 1 (`setup_cuda_env.ps1`) - fastest
- **Prefer GUI tools?** Use Method 3 (Developer Command Prompt)

## Demo Scripts

- **`hello_cuda.py`** - Ultra-minimal: Square 10 numbers (20 lines)
- **`minimal_cuda.py`** - Minimal: Double 5 numbers with clear pattern
- **`simplest_cuda_demo.py`** - Simple: Vector addition with verification (1000 elements)
- **`app.py`** - Full application: Real-time GPU image processing

## Troubleshooting

### "Cannot find compiler 'cl.exe'"
→ Use `.\run_with_msvc.ps1` or run `. .\setup_cuda_env.ps1` first

### "ImportError: No module named 'pycuda'"
→ Run `pip install pycuda` (or use Python 3.10)

### "CUDA_ERROR_NO_DEVICE"
→ Check GPU driver: `nvidia-smi`

### "cudaErrorInsufficientDriver"
→ Update GPU driver from nvidia.com

## Keyboard Shortcuts (app.py)

- **Right Arrow** - Next filter
- **Left Arrow** - Previous filter  
- **ESC** - Exit application

## Performance Tips

- Use 640x480 resolution for best FPS
- Close other GPU-intensive applications
- Monitor GPU usage: `nvidia-smi -l 1`
- Check GPU memory: `nvidia-smi --query-gpu=memory.used --format=csv`

## Getting Help

1. Check README.md for detailed troubleshooting
2. Run diagnostic: `python check_cuda_setup.py`
3. Verify setup:
   ```powershell
   python --version
   nvcc --version
   nvidia-smi
   ```
4. Report issues with full error message and system info

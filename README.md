# CUDA Real-Time Image Processing Application

A high-performance real-time image processing application that leverages CUDA GPU acceleration to apply various visual filters to live webcam feeds. This project demonstrates the power of parallel computing for computer vision applications, achieving smooth real-time performance through custom CUDA kernels.

## Overview

This application captures video from your webcam and applies sophisticated image processing filters in real-time using GPU acceleration. The implementation uses PyCUDA to write custom CUDA kernels that process images directly on the GPU, significantly outperforming CPU-based alternatives.

## Features

### Real-Time Processing
- Live webcam capture and display at 30 FPS
- GPU-accelerated image processing with CUDA kernels
- Minimal latency between capture and display
- Interactive filter switching without performance degradation

### Image Filters
The application includes 11 different visual effects:

**Basic Filters:**
- Original (no processing)
- Grayscale Conversion
- Color Negative
- Sepia Tone

**Advanced Effects:**
- Edge Detection (Sobel operator)
- Gaussian Blur (5x5 kernel)
- Emboss Effect
- Pencil Sketch
- Bilateral Filter (noise reduction)
- Cartoon Effect (color quantization)
- Vignette Effect (vintage-style darkening)

### User Interface
- Clean PyGame-based interface
- Real-time filter name display
- Keyboard controls for navigation
- Visual instructions overlay

## Technical Implementation

### CUDA Kernels
Each filter is implemented as a custom CUDA kernel optimized for parallel execution. The kernels operate directly on image pixel data, with each thread processing individual pixels or small neighborhoods for convolution operations.

### Memory Management
- Efficient GPU memory allocation for input and output buffers
- Optimized data transfer between CPU and GPU
- Proper cleanup and memory deallocation

### Performance Optimizations
- 16x16 thread block configuration for optimal GPU utilization
- Contiguous memory layouts for efficient data access
- Minimized CPU-GPU data transfers

## Installation

### Hardware Requirements

**NVIDIA GPU (Required)**
- NVIDIA GPU with compute capability 3.0 or higher
- Minimum 2GB VRAM (4GB+ recommended for higher resolutions)
- Supported GPU families:
  - GeForce GTX 600 series or newer
  - GeForce RTX series (all models)
  - Quadro K series or newer
  - Tesla K series or newer

**System Requirements**
- **RAM**: Minimum 8GB (16GB recommended)
- **CPU**: Multi-core processor (Intel i5/AMD Ryzen 5 or equivalent)
- **Storage**: 2GB free space for dependencies
- **Camera**: USB webcam or integrated camera (minimum 480p resolution)

**Operating System Support**
- Windows 10/11 (64-bit)
- Ubuntu 18.04+ (64-bit)
- macOS 10.14+ (Intel Macs with eGPU or Apple Silicon with GPU acceleration)

### Software Prerequisites
- **NVIDIA GPU Driver**: Latest stable driver (version 450.80.02 or newer)
- **CUDA Toolkit**: Version 10.2 or newer (12.x recommended)
- **Python**: 3.8 to 3.11 (3.12+ may have compatibility issues with PyCUDA)
- **Webcam**: Any USB Video Class (UVC) compatible camera

### Compatibility Check

Before installation, verify your GPU compatibility:

**Windows:**
```powershell
nvidia-smi
```

**Linux/macOS:**
```bash
nvidia-smi
lspci | grep -i nvidia  # Linux only
```

**Check CUDA compatibility:**
```bash
nvcc --version
```

**Minimum GPU Memory Test:**
Your GPU should have at least 2GB VRAM. For 1080p processing, 4GB+ is recommended.

### Windows Installation Challenges

Installing CUDA on Windows requires careful attention to version compatibility. The most common issue is PyCUDA build failures with newer Visual Studio versions.

#### **Known Working Combinations:**
- Python 3.10 + CUDA 12.4 + VS 2022 (MSVC 14.38 or earlier)
- Python 3.9 + CUDA 11.8 + VS 2019/2022
- Python 3.8 + CUDA 11.x + VS 2019

#### **Problem: PyCUDA Build Failure on Windows**

If you see errors like:
```
Unknown compiler version - please run the configure tests
error C2734: 'const' object must be initialized
error C2975: invalid template argument
Failed building wheel for pycuda
```

This occurs because PyCUDA's bundled Boost library is incompatible with MSVC 14.44+ (VS 2022 latest updates).

#### **Solution Options (Choose One):**

**Option 1: Use Python 3.10 (Recommended - Fastest)**

Pre-built wheels are available for Python 3.10:

```powershell
# Check if Python 3.10 is installed
py -3.10 --version

# If not installed, download from python.org
# Then create a new virtual environment:
py -3.10 -m venv .venv310
.venv310\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**Option 2: Install from Pre-built Wheel (Python 3.11)**

Download a compatible wheel from Christoph Gohlke's collection:
1. Visit: https://github.com/cgohlke/pycuda-build/releases
2. Download matching your Python version and CUDA toolkit
3. Install:

```powershell
pip install --upgrade pip numpy
pip install pycuda-2024.1+cuda126-cp311-cp311-win_amd64.whl
```

**Option 3: Use Conda (Most Reliable)**

Conda handles dependencies automatically:

```powershell
# Install Miniconda if not already installed
# Download from: https://docs.conda.io/en/latest/miniconda.html

conda create -n cuda-app python=3.10
conda activate cuda-app
conda install -c conda-forge pycuda
pip install opencv-python pygame numpy
```

**Option 4: Use WSL2 (Linux Environment on Windows)**

Most reliable for CUDA development:

```powershell
# Install WSL2 (one-time setup)
wsl --install -d Ubuntu

# Inside Ubuntu WSL:
sudo apt update
sudo apt install nvidia-cuda-toolkit python3-dev python3-pip python3-venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option 5: Downgrade CUDA Toolkit**

If you have CUDA 12.6, try CUDA 12.4 or 11.8:
1. Uninstall current CUDA Toolkit
2. Download older version from NVIDIA archives
3. Install and update PATH
4. Retry: `pip install pycuda`

**Option 6: Build from Source (Advanced)**

Only for experienced users:

```powershell
# Install full Boost (not just PyCUDA's subset)
# Download from: https://www.boost.org/

# Set environment variables
$env:BOOST_ROOT = "C:\local\boost_1_84_0"
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"

# Clone and build
git clone https://github.com/inducer/pycuda.git
cd pycuda
python configure.py --cuda-root="$env:CUDA_PATH" --boost-root="$env:BOOST_ROOT"
pip install -e .
```

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/LiteObject/CUDA-Image-Processing-App.git
   cd CUDA-Image-Processing-App
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Windows Users: Setup CUDA Environment First

Before running any CUDA scripts on Windows, you need to activate the Visual Studio compiler environment:

**Option A: One-time setup per PowerShell session (Recommended)**
```powershell
# Run this once when you open PowerShell
. .\setup_cuda_env.ps1

# Then run your scripts normally
python hello_cuda.py
python app.py
```

**Option B: Use the helper script (No setup needed)**
```powershell
# Run scripts with MSVC environment automatically
.\run_with_msvc.ps1 hello_cuda.py
.\run_with_msvc.ps1 app.py
```

### Linux/macOS Users

```bash
python app.py
```

### Controls
- **Right Arrow**: Switch to next filter
- **Left Arrow**: Switch to previous filter
- **ESC**: Exit application

The application will automatically detect and use your default webcam. The current filter name is displayed in the top-left corner, and control instructions appear at the bottom of the window.

## PowerShell Helper Scripts (Windows Only)

### Why Are These Scripts Necessary?

On Windows, running CUDA programs requires the Visual Studio C++ compiler (`cl.exe`) to be in your PATH. However, Visual Studio doesn't add its compilers to the system PATH by default to avoid conflicts between multiple installed versions.

**The Problem:**
```
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

**The Solution:**  
Two PowerShell scripts that automatically configure the Visual Studio compiler environment for you.

### Script Comparison

| Feature | `setup_cuda_env.ps1` | `run_with_msvc.ps1` |
|---------|---------------------|-------------------|
| **Setup Frequency** | Once per PowerShell session | Every script run |
| **Usage** | `. .\setup_cuda_env.ps1` then `python script.py` | `.\run_with_msvc.ps1 script.py` |
| **Environment Duration** | Entire PowerShell session | Single script execution |
| **Best For** | Active development, multiple runs | Quick single runs, beginners |
| **Speed** | Fast (setup once, run many) | Slightly slower (setup each time) |
| **Ease of Use** | Medium (must remember to run first) | Easy (one command) |
| **Virtual Environment** | Manual activation | Auto-activates |

### Detailed Explanation

**`setup_cuda_env.ps1` - Session Setup**
```powershell
# Run once when you open PowerShell (note the dot-space)
. .\setup_cuda_env.ps1

# Now cl.exe is available for this entire session
python hello_cuda.py
python app.py
# ... run as many scripts as you want
```

**How it works:**
1. Locates Visual Studio Build Tools 2022
2. Runs `vcvars64.bat` (Microsoft's environment setup script)
3. Captures and imports all environment variables into PowerShell
4. Verifies `cl.exe` is now accessible

**Use this when:**
- You're developing and will run multiple scripts
- You want the fastest execution for repeated runs
- You understand PowerShell environment concepts

**`run_with_msvc.ps1` - Per-Script Wrapper**
```powershell
# No setup needed - just run
.\run_with_msvc.ps1 hello_cuda.py
.\run_with_msvc.ps1 app.py
```

**How it works:**
1. Creates a temporary batch file
2. Sets up Visual Studio environment in that batch file
3. Activates your virtual environment (if present)
4. Runs your Python script
5. Cleans up (environment changes don't persist)

**Use this when:**
- You're just trying CUDA for the first time
- You only need to run one script
- You want the simplest possible command
- You're sharing instructions with others

### Alternative: Developer Command Prompt

Instead of using these scripts, you can use Visual Studio's pre-configured command prompt:

1. Start Menu → Search for "Developer Command Prompt for VS 2022"
2. Navigate to your project directory
3. Activate virtual environment: `.venv\Scripts\activate`
4. Run scripts normally: `python hello_cuda.py`

### Technical Background

**Why Windows is Different:**  
Linux typically has compilers in the system PATH (`/usr/bin/gcc`), so CUDA works immediately. Windows keeps Visual Studio compilers in versioned directories to support multiple installations, requiring explicit environment configuration.

**What `vcvars64.bat` Does:**  
Microsoft's script that sets up:
- Compiler paths (adds `cl.exe` to PATH)
- Include directories for headers
- Library paths for linking
- Architecture-specific settings

## Project Structure

```
CUDA-Image-Processing-App/
├── app.py                    # Main application - real-time GPU image processing
├── check_cuda_setup.py       # Diagnostic tool - verify CUDA environment
├── hello_cuda.py             # Tutorial 1 - Ultra-minimal (squares 10 numbers)
├── minimal_cuda.py           # Tutorial 2 - Minimal (doubles 5 numbers)
├── simplest_cuda_demo.py     # Tutorial 3 - Simple (vector addition + verification)
├── setup_cuda_env.ps1        # Windows: Sets up VS compiler for entire PowerShell session
├── run_with_msvc.ps1         # Windows: Runs scripts with VS compiler (no setup needed)
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── QUICKSTART.md             # Quick reference guide
└── docs/                     # CUDA learning materials
    ├── cuda-basics.md
    ├── cuda-execution-flow.md
    ├── cuda-memory-hierarchy.md
    └── cuda-program-steps.md
```

## Learning Path

If you're new to CUDA, follow this progression:

**1. Learn the Basics** 📚
- Read `docs/cuda-basics.md` for foundational concepts
- Understand the execution model in `docs/cuda-execution-flow.md`

**2. Run Tutorial Scripts** 🎓
```powershell
# Start with the ultra-minimal example (20 lines)
python hello_cuda.py

# Move to the minimal example (shows data transfer pattern)
python minimal_cuda.py

# Try the simple example (production-style patterns)
python simplest_cuda_demo.py
```

**3. Run the Full Application** 🚀
```powershell
# Real-time GPU image processing with 11 filters
python app.py
```

**4. Troubleshoot Issues** 🔧
```powershell
# Comprehensive environment diagnostics
python check_cuda_setup.py
```

## Dependencies

- **OpenCV**: Video capture and basic image operations
- **PyCUDA**: CUDA kernel compilation and GPU memory management
- **NumPy**: Numerical array operations and data handling
- **PyGame**: Real-time display and user interface

## Performance

This GPU-accelerated implementation provides significant performance improvements over CPU-based image processing:

- Real-time processing at 30 FPS for 640x480 resolution
- Parallel processing of thousands of pixels simultaneously
- Low-latency filter switching
- Efficient memory utilization

## Troubleshooting

### PyCUDA Installation Issues

#### **Critical: Windows Build Failures with MSVC 14.44+**

**Symptoms:**
```
Unknown compiler version - please run the configure tests
error C2734: 'const' object must be initialized
error C2975: invalid template argument for 'pycudaboost::mpl::if_c'
Failed building wheel for pycuda
```

**Root Cause:**  
PyCUDA's bundled Boost subset (circa 2019) is incompatible with Visual Studio 2022's latest compiler (MSVC 14.44+). The old Boost code uses C++ patterns that newer compilers reject.

**Quick Fix (Choose One):**

1. **Downgrade to Python 3.10** (Has pre-built wheels):
   ```powershell
   py -3.10 -m venv .venv310
   .venv310\Scripts\Activate.ps1
   pip install numpy pycuda opencv-python pygame
   ```

2. **Use Conda** (Handles compilation):
   ```powershell
   conda create -n cuda-app python=3.10 pycuda -c conda-forge
   conda activate cuda-app
   pip install opencv-python pygame
   ```

3. **Download Pre-built Wheel**:
   - Visit: https://github.com/cgohlke/pycuda-build/releases
   - Download wheel matching your Python/CUDA version
   - Install: `pip install pycuda-2024.1+cuda126-cp311-cp311-win_amd64.whl`

4. **Use WSL2** (Recommended for serious CUDA development):
   ```bash
   wsl --install -d Ubuntu
   # Inside Ubuntu:
   sudo apt install nvidia-cuda-toolkit python3-dev python3-venv
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

**Problem: CUDA Toolkit not found**
```
nvcc not found in PATH
```

**Solution:**
1. Download CUDA Toolkit from NVIDIA (version 12.4 or 11.8 recommended)
2. Add to system PATH:
   ```powershell
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
   # Make permanent in System Properties → Environment Variables
   ```
3. Verify: `nvcc --version`

**Problem: nvcc can't find compiler 'cl.exe'**
```
nvcc fatal : Cannot find compiler 'cl.exe' in PATH
```

**Root Cause:**  
NVCC needs Visual Studio's C++ compiler, but it's not in your PATH by default.

**Solution:**

Use one of the provided helper scripts:

**Quick Run (No setup):**
```powershell
.\run_with_msvc.ps1 hello_cuda.py
.\run_with_msvc.ps1 app.py
```

**One-time setup per session:**
```powershell
# Run once to set up environment
. .\setup_cuda_env.ps1

# Then run scripts normally
python hello_cuda.py
python app.py
```

**Manual (Alternative):**
```powershell
# Open "Developer Command Prompt for VS 2022" from Start Menu
# Navigate to project directory
cd C:\Users\Owner\source\repos\LiteObject\CUDA-Image-Processing-App
.venv\Scripts\activate
python hello_cuda.py
```

### OpenCV Issues

**Problem: Camera not detected**
```
Error: Could not open camera
```

**Solutions:**
1. Check camera permissions in Windows Settings
2. Ensure no other application is using the camera
3. Try different camera indices:
   ```python
   cap = cv2.VideoCapture(1)  # Try index 1, 2, etc.
   ```

**Problem: OpenCV installation with CUDA support**

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-contrib-python
```

### Runtime Errors

**Problem: CUDA out of memory**
```
pycuda._driver.MemoryError: cuMemAlloc failed: out of memory
```

**Solutions:**
1. Reduce image resolution in the code
2. Close other GPU-intensive applications
3. Check available GPU memory:
   ```python
   import pycuda.driver as cuda
   cuda.mem_get_info()
   ```

**Problem: Slow performance or low FPS**

**Solutions:**
1. Check if using integrated vs dedicated GPU
2. Ensure CUDA drivers are up to date
3. Monitor GPU utilization with `nvidia-smi`
4. Reduce camera resolution for better performance

### Installation Method Comparison

| Method | Pros | Cons | Best For |
|--------|------|------|----------|
| **Python 3.10** | Pre-built wheels, fast setup | Older Python version | Quick start, learning |
| **Conda** | Handles all deps, reliable | Large download (~2GB) | Production, stability |
| **WSL2** | Native Linux, best compatibility | Extra setup step | Serious development |
| **Pre-built Wheel** | Works with Python 3.11 | Manual download | Specific requirements |
| **Build from Source** | Latest code, customizable | Complex, time-consuming | Advanced users only |

### Environment Issues

**Problem: Virtual environment activation fails**

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.venv\Scripts\Activate.ps1
```

**Windows Command Prompt:**
```cmd
.venv\Scripts\activate.bat
```

**Problem: Missing Visual C++ Redistributables**

**Solution:**
Download and install Microsoft Visual C++ Redistributable packages from Microsoft's website (both x86 and x64 versions).

**Problem: "ImportError: DLL load failed" when importing PyCUDA**

**Solutions:**
1. Ensure CUDA Toolkit bin directory is in PATH
2. Install matching Visual C++ Redistributables
3. Check CUDA driver version matches toolkit:
   ```powershell
   nvidia-smi  # Shows driver CUDA version
   nvcc --version  # Shows toolkit version
   ```
4. If mismatch, update GPU driver from nvidia.com

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module named 'pycuda'` | PyCUDA not installed | Follow PyCUDA installation steps |
| `pygame.error: No available video device` | Display/graphics issue | Install/update graphics drivers |
| `cv2.error: function not implemented` | OpenCV compiled without feature | Install `opencv-contrib-python` |
| `CUDA_ERROR_NO_DEVICE` | No CUDA-capable GPU | Check GPU compatibility |

### Why is CUDA Installation Challenging on Windows?

Unlike Linux where `apt install nvidia-cuda-toolkit python3-pycuda` often "just works," Windows installation faces several challenges:

**1. Compiler Version Incompatibility**
- PyCUDA bundles old Boost C++ library (2019-era code)
- Microsoft updates MSVC frequently with breaking changes
- New compilers reject old C++ patterns
- Result: Build failures with latest Visual Studio 2022

**2. No Universal Binary Wheels**
- Linux: Pre-compiled for most configurations
- Windows: Limited wheels for specific Python/CUDA combinations
- Missing wheel → forced source build → compilation errors

**3. Complex Dependency Chain**
```
Your App → PyCUDA → CUDA Toolkit → GPU Driver → Windows SDK → Visual Studio Build Tools
```
Each link must be version-compatible with neighbors.

**4. PATH Environment Hell**
- Multiple CUDA versions can coexist
- Visual Studio paths can conflict
- Wrong `nvcc` or compiler gets picked first
- Changes require new shell to take effect

**5. Driver vs. Toolkit Mismatches**
```
GPU Driver: CUDA 12.6 (from driver update)
CUDA Toolkit: 12.4 (what you installed)
PyCUDA: Built for 12.2 (from old wheel)
→ Runtime errors
```

**Linux Advantage:**
- System package manager resolves dependencies
- GCC compiler is stable across versions
- Standard library locations
- Better error messages
- CUDA ecosystem primarily targets Linux

**Recommended Approach for Windows Users:**
1. **Learning/Hobby**: Use Python 3.10 with pre-built wheels
2. **Production**: Use Conda for dependency management
3. **Serious Development**: Use WSL2 for Linux-like experience
4. **Enterprise**: Docker containers with pre-configured CUDA

### Getting Help

If you continue to experience issues:

1. **Run the diagnostic tool:**
   ```powershell
   python check_cuda_setup.py
   ```
   This will check Python version, packages, GPU driver, CUDA toolkit, MSVC compiler, and device access.

2. **Verify hardware compatibility:**
   - NVIDIA GPU with compute capability 3.0+ (check with `nvidia-smi`)
   - Minimum 2GB VRAM available
   - Latest NVIDIA drivers installed
   - CUDA Toolkit properly configured

3. **Check software versions:**
   - Python 3.8-3.11 (avoid 3.12+ for now)
   - Compatible PyCUDA version for your CUDA toolkit
   - Updated OpenCV with video support

3. **Enable debug output:**
   ```python
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   ```

4. **Document your configuration:**
   ```powershell
   python --version
   nvcc --version
   nvidia-smi
   pip list | findstr "pycuda numpy opencv"
   ```

5. **Report issues on GitHub** with:
   - Full error message
   - System specifications (OS, GPU, CUDA version)
   - Python version and package versions
   - Installation method attempted

## Development

### Adding New Filters
To add a new filter:

1. Implement the CUDA kernel in the `compile_kernels()` method
2. Add the kernel function reference in the same method
3. Update the `filters` list with the new filter name
4. Add the filter case in the `apply_filter()` method

### Customization
- Modify kernel parameters for different effects
- Adjust thread block sizes for different GPU architectures
- Change camera resolution in the initialization code
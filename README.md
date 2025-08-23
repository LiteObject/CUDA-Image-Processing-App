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

### Prerequisites
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- Python 3.8 or higher
- Webcam or video capture device

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

Run the application:
```bash
python app.py
```

### Controls
- **Right Arrow**: Switch to next filter
- **Left Arrow**: Switch to previous filter
- **ESC**: Exit application

The application will automatically detect and use your default webcam. The current filter name is displayed in the top-left corner, and control instructions appear at the bottom of the window.

## Project Structure

```
CUDA-Image-Processing-App/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
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

**Problem: PyCUDA compilation fails with Visual Studio 2022**
```
Unknown compiler version - please run the configure tests and report the results
error C2988: unrecognizable template declaration/definition
```

**Solutions:**
1. **Use pre-built wheels (Recommended):**
   ```bash
   pip install --upgrade pip
   pip install pycuda --find-links https://christophergohlke.github.io/pythonlibs/
   ```

2. **Install older PyCUDA version:**
   ```bash
   pip install pycuda==2019.1.2
   ```

3. **Use conda instead of pip:**
   ```bash
   conda install -c conda-forge pycuda
   ```

4. **Update Visual Studio Build Tools:**
   - Download and install the latest Visual Studio Build Tools
   - Ensure C++ build tools are included

**Problem: CUDA Toolkit not found**
```
nvcc not found in PATH
```

**Solution:**
1. Download and install CUDA Toolkit from NVIDIA's website
2. Add CUDA to your system PATH:
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
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
Download and install Microsoft Visual C++ Redistributable packages from Microsoft's website.

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ImportError: No module named 'pycuda'` | PyCUDA not installed | Follow PyCUDA installation steps |
| `pygame.error: No available video device` | Display/graphics issue | Install/update graphics drivers |
| `cv2.error: function not implemented` | OpenCV compiled without feature | Install `opencv-contrib-python` |
| `CUDA_ERROR_NO_DEVICE` | No CUDA-capable GPU | Check GPU compatibility |

### Getting Help

If you continue to experience issues:

1. **Check system requirements:**
   - NVIDIA GPU with compute capability 3.0+
   - Latest NVIDIA drivers
   - Python 3.8-3.11 (3.12+ may have compatibility issues)

2. **Enable debug output:**
   ```python
   import os
   os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
   ```

3. **Report issues on GitHub** with:
   - Full error message
   - System specifications (OS, GPU, CUDA version)
   - Python version and package versions

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
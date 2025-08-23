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
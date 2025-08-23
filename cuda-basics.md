# CUDA Programming Basics

## Introduction to CUDA

CUDA (Compute Unified Device Architecture) is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use NVIDIA GPUs for general-purpose computing, not just graphics rendering. This guide covers the fundamental concepts you need to understand CUDA programming.

## Why Use CUDA?

Traditional CPUs have a few powerful cores optimized for sequential processing. In contrast, GPUs have thousands of smaller cores designed for parallel processing. This makes GPUs excellent for tasks that can be divided into many independent operations, such as image processing, mathematical computations, and machine learning.

### Performance Benefits
- Massive parallelism: Execute thousands of threads simultaneously
- High memory bandwidth: Fast data access for parallel operations
- Specialized hardware: Optimized for floating-point calculations

## Core Concepts

### Host vs Device
- **Host**: The CPU and its memory (RAM)
- **Device**: The GPU and its memory (VRAM)

CUDA programs run on both the host and device, with the host controlling execution and the device performing parallel computations.

### Kernels
A kernel is a function that runs on the GPU. When you launch a kernel, it executes across many GPU threads in parallel. Each thread typically processes a small portion of the data.

```python
# CUDA kernel written as a string in Python
kernel_source = '''
__global__ void simple_kernel(int *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * 2;
    }
}
'''

# Compile the kernel using PyCUDA
from pycuda.compiler import SourceModule
mod = SourceModule(kernel_source)
simple_kernel = mod.get_function("simple_kernel")
```

The `__global__` keyword indicates this function runs on the device but is called from the host.

### Thread Hierarchy
CUDA organizes threads in a hierarchical structure:

1. **Thread**: The smallest unit of execution
2. **Block**: A group of threads that can cooperate and share memory
3. **Grid**: A collection of blocks

This hierarchy helps organize parallel work and optimize memory access patterns.

### Memory Types
CUDA devices have several types of memory:

- **Global Memory**: Large but slow, accessible by all threads
- **Shared Memory**: Fast but small, shared within a block
- **Local Memory**: Private to each thread
- **Constant Memory**: Read-only, cached for fast access

## Basic CUDA Workflow with PyCUDA

### 1. Memory Allocation
First, allocate memory on both host and device:

```python
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# Host memory (NumPy array)
size = 1024
h_data = np.random.randint(0, 10, size).astype(np.int32)

# Device memory
d_data = cuda.mem_alloc(h_data.nbytes)
```

### 2. Data Transfer
Copy data between host and device:

```python
# Host to device
cuda.memcpy_htod(d_data, h_data)

# Device to host (after processing)
result = np.empty_like(h_data)
cuda.memcpy_dtoh(result, d_data)
```

### 3. Kernel Launch
Execute the kernel with specified grid and block dimensions:

```python
block_size = 256
grid_size = (size + block_size - 1) // block_size

simple_kernel(d_data, np.int32(size), 
              block=(block_size, 1, 1), 
              grid=(grid_size, 1))
```

### 4. Synchronization
Wait for the kernel to complete:

```python
cuda.Context.synchronize()
```

### 5. Cleanup
Free allocated memory:

```python
d_data.free()
```

## Thread Indexing

Understanding how to calculate thread indices is crucial for CUDA programming. Each thread needs to know which data element it should process.

### 1D Indexing
For one-dimensional data:
```python
kernel_1d = '''
__global__ void process_1d(float *data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        data[idx] = data[idx] * 2.0f;
    }
}
'''
```

### 2D Indexing
For two-dimensional data like images:
```python
kernel_2d = '''
__global__ void process_image(unsigned char *input, unsigned char *output, 
                             int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = input[idx] * 0.5f;  // Darken pixel
    }
}
'''
```

## Common Patterns

### Image Processing
Most image processing operations follow this pattern:

```python
# Complete example of a grayscale conversion kernel
grayscale_kernel = '''
__global__ void rgb_to_grayscale(unsigned char *input, unsigned char *output, 
                                int width, int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3;  // RGB has 3 channels
        
        unsigned char r = input[idx + 2];
        unsigned char g = input[idx + 1]; 
        unsigned char b = input[idx];
        
        unsigned char gray = 0.299f * r + 0.587f * g + 0.114f * b;
        
        output[idx] = gray;
        output[idx + 1] = gray;
        output[idx + 2] = gray;
    }
}
'''

# Python code to use this kernel
import cv2
from pycuda.compiler import SourceModule

# Compile kernel
mod = SourceModule(grayscale_kernel)
grayscale_func = mod.get_function("rgb_to_grayscale")

# Process an image
image = cv2.imread("input.jpg")
height, width = image.shape[:2]

# Allocate GPU memory
d_input = cuda.mem_alloc(image.nbytes)
d_output = cuda.mem_alloc(image.nbytes)

# Copy to GPU
cuda.memcpy_htod(d_input, image)

# Launch kernel
block_size = (16, 16, 1)
grid_size = ((width + 15) // 16, (height + 15) // 16, 1)
grayscale_func(d_input, d_output, np.int32(width), np.int32(height),
               block=block_size, grid=grid_size)

# Copy result back
result = np.empty_like(image)
cuda.memcpy_dtoh(result, d_output)
```

### Reduction Operations
When combining many values into one (like sum or maximum):

```python
reduction_kernel = '''
__global__ void sum_reduction(float *input, float *output, int size) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Load data into shared memory
    shared_data[tid] = (idx < size) ? input[idx] : 0;
    __syncthreads();
    
    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}
'''
```

## Memory Optimization

### Coalesced Memory Access
Arrange data access patterns so consecutive threads access consecutive memory locations. This maximizes memory bandwidth utilization.

### Shared Memory Usage
Use shared memory for data that multiple threads in a block need to access. This is much faster than global memory but requires careful synchronization.

### Occupancy
Maximize the number of active threads on the GPU by choosing appropriate block sizes and minimizing resource usage per thread.

## Error Handling

Always check for CUDA errors to debug issues:

```python
import pycuda.driver as cuda

try:
    # CUDA operations
    cuda.memcpy_htod(d_data, h_data)
    kernel_func(d_data, block=(256, 1, 1), grid=(grid_size, 1))
    cuda.Context.synchronize()
    
except cuda.Error as e:
    print(f"CUDA error: {e}")
    
# Alternative: Check for errors after each operation
def check_cuda_error(operation_name):
    try:
        cuda.Context.synchronize()
    except cuda.Error as e:
        print(f"CUDA error in {operation_name}: {e}")
        raise

# Usage
cuda.memcpy_htod(d_data, h_data)
check_cuda_error("memory copy")
```

## Best Practices

### Algorithm Design
- Identify parallel portions of your algorithm
- Minimize data dependencies between threads
- Balance workload across threads

### Memory Management
- Minimize host-device transfers
- Use appropriate memory types for different access patterns
- Consider using unified memory for simpler programming

### Performance Optimization
- Profile your code to identify bottlenecks
- Experiment with different block sizes
- Optimize memory access patterns
- Use streams for overlapping computation and data transfer

## Common Pitfalls

### Race Conditions
Multiple threads accessing the same memory location without proper synchronization can lead to unpredictable results.

### Divergent Branches
When threads in the same warp take different execution paths, performance suffers. Try to minimize conditional statements in kernels.

### Memory Bandwidth Limits
Even with perfect parallelization, memory bandwidth often becomes the limiting factor. Focus on optimizing memory access patterns.

## Learning Path

1. **Start Simple**: Begin with basic kernels that process one element per thread
   ```python
   # Simple element-wise operation
   simple_kernel = '''
   __global__ void add_arrays(float *a, float *b, float *result, int size) {
       int idx = threadIdx.x + blockIdx.x * blockDim.x;
       if (idx < size) {
           result[idx] = a[idx] + b[idx];
       }
   }
   '''
   ```

2. **Understand Indexing**: Master thread indexing for 1D and 2D problems

3. **Memory Management**: Learn to efficiently transfer and access data using PyCUDA's memory functions

4. **Optimization**: Study memory coalescing and occupancy

5. **Advanced Topics**: Explore shared memory, streams, and multi-GPU programming

## Complete PyCUDA Example

Here's a full working example that demonstrates the concepts:

```python
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Define the kernel
kernel_code = '''
__global__ void vector_add(float *a, float *b, float *result, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        result[idx] = a[idx] + b[idx];
    }
}
'''

def main():
    # Compile the kernel
    mod = SourceModule(kernel_code)
    vector_add = mod.get_function("vector_add")
    
    # Create test data
    n = 1024
    a = np.random.randn(n).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)
    result = np.zeros_like(a)
    
    # Allocate GPU memory
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    result_gpu = cuda.mem_alloc(result.nbytes)
    
    # Copy data to GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)
    
    # Launch kernel
    block_size = 256
    grid_size = (n + block_size - 1) // block_size
    
    vector_add(a_gpu, b_gpu, result_gpu, np.int32(n),
               block=(block_size, 1, 1),
               grid=(grid_size, 1))
    
    # Copy result back
    cuda.memcpy_dtoh(result, result_gpu)
    
    # Verify result
    expected = a + b
    print(f"Results match: {np.allclose(result, expected)}")
    
    # Cleanup
    a_gpu.free()
    b_gpu.free()
    result_gpu.free()

if __name__ == "__main__":
    main()
```

# Conclusion
CUDA programming requires thinking differently about algorithms and data organization. With practice, you'll develop intuition for parallel problem-solving and be able to harness the full power of modern GPUs for high-performance computing applications.

"""hello_cuda.py
Ultra-minimal CUDA demo: square 10 numbers on the GPU.

Usage (after installing dependencies):
    pip install pycuda numpy
    python hello_cuda.py

Notes:
 - This file intentionally keeps logic minimal for teaching.
 - Imports are wrapped to provide a clear message if dependencies are missing.
"""

# pylint: disable=import-error
try:
    import numpy as np
    import pycuda.autoinit  # pylint: disable=unused-import  # Creates a CUDA context automatically
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover - only triggers when deps missing
    raise SystemExit(
        "Missing dependency for CUDA demo. Install with: pip install numpy pycuda"
    ) from exc

# GPU kernel: each thread squares one element
KERNEL_SRC = """
__global__ void square(float *data) {
    int i = threadIdx.x;   // Only using 1 block in this toy example
    data[i] = data[i] * data[i];
}
"""

mod = SourceModule(KERNEL_SRC)
square = mod.get_function("square")

# Host data
h_data = np.arange(10, dtype=np.float32)
print("Input :", h_data)

# Device allocation & copy
d_data = cuda.mem_alloc(h_data.nbytes)
cuda.memcpy_htod(d_data, h_data)

# Launch kernel with exactly 10 threads (1 block)
square(d_data, block=(10, 1, 1), grid=(1, 1))

# Copy back
cuda.memcpy_dtoh(h_data, d_data)
print("Output:", h_data)

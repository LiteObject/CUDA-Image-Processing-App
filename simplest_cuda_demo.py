"""simplest_cuda_demo.py
Simple CUDA demo: element-wise addition of two vectors with verification.

Demonstrates:
  - Kernel with bounds check
  - Grid/block sizing
  - Data transfer
  - Result verification

Install:
    pip install pycuda numpy
Run:
    python simplest_cuda_demo.py
"""

from __future__ import annotations

# pylint: disable=import-error
try:
    import numpy as np
    import pycuda.autoinit  # pylint: disable=unused-import  # context side-effect
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependencies. Install with: pip install numpy pycuda"
    ) from exc

KERNEL_SRC = """
__global__ void add_arrays(const float *a, const float *b, float *out, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}
"""

mod = SourceModule(KERNEL_SRC)
add_arrays = mod.get_function("add_arrays")

N = 1000
h_a = np.random.randn(N).astype(np.float32)
h_b = np.random.randn(N).astype(np.float32)
h_out = np.empty_like(h_a)

# Allocate device buffers
d_a = cuda.mem_alloc(h_a.nbytes)
d_b = cuda.mem_alloc(h_b.nbytes)
d_out = cuda.mem_alloc(h_out.nbytes)

# Transfer to device
cuda.memcpy_htod(d_a, h_a)
cuda.memcpy_htod(d_b, h_b)

# Kernel launch configuration (kept in lowercase for teaching clarity)
# pylint: disable=invalid-name
threads_per_block = 128  # Typical power-of-two size
blocks = (N + threads_per_block - 1) // threads_per_block

# Launch kernel
add_arrays(
    d_a, d_b, d_out, np.int32(N), block=(threads_per_block, 1, 1), grid=(blocks, 1)
)

# Copy back
cuda.memcpy_dtoh(h_out, d_out)

# Verify
expected = h_a + h_b
match = np.allclose(expected, h_out)
print("Match:", match)
print("First 5 (GPU):", h_out[:5])
print("First 5 (CPU):", expected[:5])

if not match:
    raise RuntimeError("Mismatch between GPU and CPU results")

# Free (optional, context teardown will release)
d_a.free()
d_b.free()
d_out.free()

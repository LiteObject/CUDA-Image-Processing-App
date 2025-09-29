"""minimal_cuda.py
Minimal CUDA demo: doubles an array of 5 numbers.

Shows the basic host→device, kernel launch, device→host pattern.

Install:
    pip install pycuda numpy
Run:
    python minimal_cuda.py
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
__global__ void double_values(float *data) {
    int i = threadIdx.x; // Single block example
    data[i] = data[i] * 2.0f;
}
"""

mod = SourceModule(KERNEL_SRC)
double_values = mod.get_function("double_values")

h_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
print("Original:", h_data)

d_data = cuda.mem_alloc(h_data.nbytes)
cuda.memcpy_htod(d_data, h_data)

double_values(d_data, block=(5, 1, 1), grid=(1, 1))

cuda.memcpy_dtoh(h_data, d_data)
print("Doubled :", h_data)

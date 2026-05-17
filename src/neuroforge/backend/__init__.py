"""Backend abstractions.

The NumPy backend is the only executable backend today.  The API is intentionally
small so CUDA/OpenCL kernels can be introduced without leaking backend details
into the autograd graph.
"""

from neuroforge.backend.base import Backend
from neuroforge.backend.numpy_backend import NumPyBackend, default_backend

__all__ = ["Backend", "NumPyBackend", "default_backend"]

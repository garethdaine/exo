# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
"""
CUDA inference engine for NVIDIA GPUs.

This package provides the CUDA/PyTorch backend for distributed inference on NVIDIA
GPUs. It supports both NCCL (optimized GPU-to-GPU) and Gloo (CPU-based fallback)
communication backends for multi-GPU and multi-node inference.

Key components:
    - CudaEngine: The main engine class implementing InferenceEngine protocol
    - NcclDistributedGroup: NCCL-based distributed communication group
    - GlooDistributedGroup: Gloo-based distributed communication group

Target Hardware:
    - NVIDIA DGX Spark (Blackwell GB10)
    - NVIDIA RTX 30/40 series, A100, H100
    - Linux (x86_64 and aarch64) and Windows

Usage:
    from exo.worker.engines.cuda import CudaEngine, get_cuda_engine

    engine = get_cuda_engine()
    # or
    engine = CudaEngine()

Note:
    Type checking is relaxed in this module because PyTorch is an optional
    dependency that may not be installed on all platforms (e.g., macOS).
"""

from typing import TYPE_CHECKING

# Lazy imports to avoid ImportError on platforms without PyTorch
if TYPE_CHECKING:
    from exo.worker.engines.cuda.distributed import (
        GlooDistributedGroup,
        NcclDistributedGroup,
    )
    from exo.worker.engines.cuda.engine import CudaEngine

__all__ = [
    "CudaEngine",
    "NcclDistributedGroup",
    "GlooDistributedGroup",
    "get_cuda_engine",
    "is_cuda_available",
]


def is_cuda_available() -> bool:
    """
    Check if CUDA is available on this system.

    Returns:
        True if PyTorch is installed and CUDA is available.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_cuda_engine() -> "CudaEngine":
    """
    Get an instance of the CUDA inference engine.

    Returns:
        CudaEngine instance.

    Raises:
        ImportError: If PyTorch is not installed.
        RuntimeError: If CUDA is not available.
    """
    if not is_cuda_available():
        raise RuntimeError(
            "CUDA is not available. Ensure PyTorch is installed with CUDA support."
        )

    from exo.worker.engines.cuda.engine import CudaEngine

    return CudaEngine()

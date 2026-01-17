# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUnusedImport=false
"""
CUDA utility functions for device management and initialization.

This module provides utilities for:
- CUDA device detection and selection
- GPU memory management
- CUDA context initialization
- Device capability queries

Usage:
    from exo.worker.engines.cuda.utils_cuda import (
        get_cuda_device_count,
        get_cuda_device_info,
        initialize_cuda_device,
    )

    # Check available devices
    num_devices = get_cuda_device_count()

    # Get device info
    info = get_cuda_device_info(device_id=0)
    print(f"Device: {info['name']}, Memory: {info['total_memory_gb']:.2f} GB")
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CudaDeviceInfo:
    """Information about a CUDA device."""

    device_id: int
    name: str
    compute_capability: tuple[int, int]
    total_memory_bytes: int
    total_memory_gb: float
    is_available: bool


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


def get_cuda_device_count() -> int:
    """
    Get the number of available CUDA devices.

    Returns:
        Number of CUDA devices, or 0 if CUDA is not available.
    """
    if not is_cuda_available():
        return 0

    import torch

    return torch.cuda.device_count()


def get_cuda_device_info(device_id: int = 0) -> CudaDeviceInfo:
    """
    Get detailed information about a CUDA device.

    Args:
        device_id: The CUDA device index.

    Returns:
        CudaDeviceInfo with device details.

    Raises:
        RuntimeError: If CUDA is not available or device_id is invalid.
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available")

    import torch

    if device_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid device_id {device_id}. Only {torch.cuda.device_count()} devices available."
        )

    props = torch.cuda.get_device_properties(device_id)
    total_memory = props.total_memory

    return CudaDeviceInfo(
        device_id=device_id,
        name=props.name,
        compute_capability=(props.major, props.minor),
        total_memory_bytes=total_memory,
        total_memory_gb=total_memory / (1024**3),
        is_available=True,
    )


def get_all_cuda_devices() -> list[CudaDeviceInfo]:
    """
    Get information about all available CUDA devices.

    Returns:
        List of CudaDeviceInfo for each device.
    """
    num_devices = get_cuda_device_count()
    return [get_cuda_device_info(i) for i in range(num_devices)]


def initialize_cuda_device(
    device_id: int = 0,
    *,
    enable_cudnn_benchmark: bool = True,
    enable_tf32: bool = True,
) -> "torch.device":
    """
    Initialize a CUDA device for inference.

    This function sets up the CUDA device with optimal settings for inference,
    including cuDNN benchmarking and TF32 precision.

    Args:
        device_id: The CUDA device index to initialize.
        enable_cudnn_benchmark: Enable cuDNN benchmarking for faster convolutions.
        enable_tf32: Enable TensorFloat-32 for faster matrix operations on Ampere+.

    Returns:
        The torch.device object for the initialized device.

    Raises:
        RuntimeError: If CUDA is not available or device_id is invalid.
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available")

    import torch

    if device_id >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid device_id {device_id}. Only {torch.cuda.device_count()} devices available."
        )

    # Set the current device
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    # Configure cuDNN
    if enable_cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
        logger.debug(f"cuDNN benchmark enabled for device {device_id}")

    # Configure TF32 (available on Ampere and newer)
    if enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.debug(f"TF32 enabled for device {device_id}")

    # Prime the device with a small allocation
    _ = torch.zeros(1, device=device)

    device_info = get_cuda_device_info(device_id)
    logger.info(
        f"Initialized CUDA device {device_id}: {device_info.name} "
        f"({device_info.total_memory_gb:.2f} GB, CC {device_info.compute_capability[0]}.{device_info.compute_capability[1]})"
    )

    return device


def get_cuda_memory_info(device_id: int = 0) -> dict[str, Any]:
    """
    Get memory usage information for a CUDA device.

    Args:
        device_id: The CUDA device index.

    Returns:
        Dictionary with memory statistics:
            - allocated_bytes: Currently allocated memory
            - reserved_bytes: Memory reserved by the caching allocator
            - max_allocated_bytes: Peak allocated memory
            - total_bytes: Total device memory
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available")

    import torch

    device = torch.device(f"cuda:{device_id}")
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "total_bytes": torch.cuda.get_device_properties(device_id).total_memory,
    }


def clear_cuda_cache(device_id: int | None = None) -> None:
    """
    Clear the CUDA memory cache.

    This releases cached memory back to the CUDA driver, which can help
    reduce memory fragmentation and allow other processes to use GPU memory.

    Args:
        device_id: The device to clear cache for, or None for all devices.
    """
    if not is_cuda_available():
        return

    import torch

    if device_id is not None:
        with torch.cuda.device(device_id):
            torch.cuda.empty_cache()
            logger.debug(f"Cleared CUDA cache for device {device_id}")
    else:
        torch.cuda.empty_cache()
        logger.debug("Cleared CUDA cache for all devices")


def set_cuda_memory_fraction(fraction: float, device_id: int = 0) -> None:
    """
    Set the maximum memory fraction PyTorch can allocate on a device.

    This can be useful for sharing GPU resources with other processes.

    Args:
        fraction: The maximum fraction of memory to use (0.0 to 1.0).
        device_id: The CUDA device index.

    Raises:
        ValueError: If fraction is not in range [0.0, 1.0].
    """
    if not is_cuda_available():
        return

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"Memory fraction must be in [0.0, 1.0], got {fraction}")

    import torch

    torch.cuda.set_per_process_memory_fraction(fraction, device_id)
    logger.info(f"Set CUDA memory fraction to {fraction:.1%} for device {device_id}")


def get_optimal_dtype() -> "torch.dtype":
    """
    Get the optimal data type for inference on the current device.

    Returns:
        - torch.bfloat16 for Ampere+ GPUs (compute capability >= 8.0)
        - torch.float16 for older GPUs

    Raises:
        RuntimeError: If CUDA is not available.
    """
    if not is_cuda_available():
        raise RuntimeError("CUDA is not available")

    import torch

    device_info = get_cuda_device_info(0)
    major, _minor = device_info.compute_capability

    # Ampere (SM 8.0) and newer support bfloat16 efficiently
    if major >= 8:
        return torch.bfloat16

    # Older GPUs use float16
    return torch.float16


def synchronize_cuda(device_id: int | None = None) -> None:
    """
    Synchronize CUDA operations on the specified device.

    This blocks until all pending CUDA operations complete. Useful for
    accurate timing measurements and ensuring operations complete.

    Args:
        device_id: The device to synchronize, or None for current device.
    """
    if not is_cuda_available():
        return

    import torch

    if device_id is not None:
        torch.cuda.synchronize(device_id)
    else:
        torch.cuda.synchronize()

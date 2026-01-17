# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
"""
Unified platform detection for hardware accelerators.

This module provides a unified interface for detecting available hardware
accelerators across different platforms (macOS, Linux, Windows).

Supported accelerators:
- Apple Silicon with MLX (macOS)
- NVIDIA CUDA GPUs (Linux, Windows)
- AMD ROCm GPUs (future)
- Intel oneAPI (future)
- Vulkan compute (future)
- CPU-only fallback

Usage:
    from exo.worker.utils.platform_detection import (
        detect_platform,
        get_available_accelerators,
        PlatformInfo,
    )

    # Get platform info
    platform_info = detect_platform()
    print(f"Platform: {platform_info.os_name}")
    print(f"Accelerators: {[a.name for a in platform_info.accelerators]}")

    # Get available accelerators
    accelerators = get_available_accelerators()
    for acc in accelerators:
        print(f"{acc.accelerator_type}: {acc.device_name}")
"""

from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from loguru import logger

from exo.shared.types.profiling import AcceleratorType

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class AcceleratorInfo:
    """Information about a detected hardware accelerator."""

    accelerator_type: AcceleratorType
    device_index: int
    device_name: str
    device_id: str
    memory_total_bytes: int
    compute_capability: str | None = None
    driver_version: str | None = None
    is_available: bool = True


@dataclass
class PlatformInfo:
    """Complete platform information including all detected accelerators."""

    os_name: str
    os_version: str
    architecture: str
    python_version: str
    accelerators: list[AcceleratorInfo] = field(default_factory=list)
    default_accelerator_type: AcceleratorType = AcceleratorType.CPU_ONLY

    @property
    def has_gpu(self) -> bool:
        """Check if any GPU accelerator is available."""
        return any(
            acc.accelerator_type
            in (
                AcceleratorType.APPLE_SILICON,
                AcceleratorType.NVIDIA_CUDA,
                AcceleratorType.AMD_ROCM,
                AcceleratorType.VULKAN,
            )
            for acc in self.accelerators
        )

    @property
    def total_gpu_memory_bytes(self) -> int:
        """Total GPU memory across all accelerators."""
        return sum(
            acc.memory_total_bytes
            for acc in self.accelerators
            if acc.accelerator_type != AcceleratorType.CPU_ONLY
        )


def _detect_apple_silicon() -> list[AcceleratorInfo]:
    """Detect Apple Silicon accelerators on macOS."""
    if sys.platform != "darwin":
        return []

    machine = platform.machine().lower()
    if not any(chip in machine for chip in ("arm", "arm64", "aarch64")):
        return []

    # Check if MLX is available
    try:
        import mlx.core as mx

        # Get device info from MLX
        # MLX uses unified memory, so we use system memory as approximation
        import psutil

        total_memory = psutil.virtual_memory().total

        # Try to get chip name
        chip_name = "Apple Silicon"
        try:
            from subprocess import run

            result = run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                chip_name = result.stdout.strip()
        except Exception:
            pass

        return [
            AcceleratorInfo(
                accelerator_type=AcceleratorType.APPLE_SILICON,
                device_index=0,
                device_name=chip_name,
                device_id="apple-silicon-0",
                memory_total_bytes=total_memory,
                compute_capability=None,
                driver_version=mx.__version__ if hasattr(mx, "__version__") else None,
                is_available=True,
            )
        ]
    except ImportError:
        logger.debug("MLX not available on Apple Silicon")
        return []


def _detect_nvidia_cuda() -> list[AcceleratorInfo]:
    """Detect NVIDIA CUDA GPUs."""
    if sys.platform == "darwin":
        # No CUDA on macOS
        return []

    try:
        from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus

        nvidia_gpus = detect_nvidia_gpus()
        return [
            AcceleratorInfo(
                accelerator_type=AcceleratorType.NVIDIA_CUDA,
                device_index=gpu.index,
                device_name=gpu.name,
                device_id=gpu.uuid,
                memory_total_bytes=gpu.memory_total_bytes,
                compute_capability=f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}",
                driver_version=gpu.driver_version,
                is_available=True,
            )
            for gpu in nvidia_gpus
        ]
    except Exception as e:
        logger.debug(f"NVIDIA GPU detection failed: {e}")
        return []


def _detect_amd_rocm() -> list[AcceleratorInfo]:
    """Detect AMD ROCm GPUs (placeholder for future implementation)."""
    # TODO: Implement ROCm detection using rocm-smi or pyrsmi
    return []


def _detect_intel_oneapi() -> list[AcceleratorInfo]:
    """Detect Intel oneAPI accelerators (placeholder for future implementation)."""
    # TODO: Implement Intel GPU detection
    return []


def _detect_vulkan() -> list[AcceleratorInfo]:
    """Detect Vulkan-capable devices (placeholder for future implementation)."""
    # TODO: Implement Vulkan device enumeration
    return []


def detect_platform() -> PlatformInfo:
    """
    Detect the current platform and all available hardware accelerators.

    Returns:
        PlatformInfo object with complete platform information.
    """
    # Get OS information
    os_name = platform.system()
    os_version = platform.version()
    architecture = platform.machine()
    python_version = platform.python_version()

    # Detect all accelerators
    accelerators: list[AcceleratorInfo] = []

    # Apple Silicon (macOS only)
    accelerators.extend(_detect_apple_silicon())

    # NVIDIA CUDA (Linux, Windows)
    accelerators.extend(_detect_nvidia_cuda())

    # AMD ROCm (Linux)
    accelerators.extend(_detect_amd_rocm())

    # Intel oneAPI
    accelerators.extend(_detect_intel_oneapi())

    # Vulkan (cross-platform)
    accelerators.extend(_detect_vulkan())

    # Determine default accelerator type
    default_type = AcceleratorType.CPU_ONLY
    if accelerators:
        # Priority order: Apple Silicon > NVIDIA > AMD > Intel > Vulkan
        priority = {
            AcceleratorType.APPLE_SILICON: 5,
            AcceleratorType.NVIDIA_CUDA: 4,
            AcceleratorType.AMD_ROCM: 3,
            AcceleratorType.INTEL_ONEAPI: 2,
            AcceleratorType.VULKAN: 1,
            AcceleratorType.CPU_ONLY: 0,
        }
        default_type = max(
            (acc.accelerator_type for acc in accelerators),
            key=lambda t: priority.get(t, 0),
        )

    return PlatformInfo(
        os_name=os_name,
        os_version=os_version,
        architecture=architecture,
        python_version=python_version,
        accelerators=accelerators,
        default_accelerator_type=default_type,
    )


def get_available_accelerators() -> list[AcceleratorInfo]:
    """
    Get a list of all available hardware accelerators.

    Returns:
        List of AcceleratorInfo objects for available accelerators.
    """
    return detect_platform().accelerators


def get_default_accelerator() -> AcceleratorInfo | None:
    """
    Get the default/recommended accelerator for this platform.

    Returns:
        AcceleratorInfo for the default accelerator, or None if only CPU is available.
    """
    platform_info = detect_platform()
    if not platform_info.accelerators:
        return None

    # Return the first accelerator (already sorted by priority in detect_platform)
    return platform_info.accelerators[0]


def get_accelerator_by_type(
    accelerator_type: AcceleratorType,
) -> list[AcceleratorInfo]:
    """
    Get all accelerators of a specific type.

    Args:
        accelerator_type: The type of accelerator to find.

    Returns:
        List of AcceleratorInfo objects matching the type.
    """
    return [
        acc
        for acc in detect_platform().accelerators
        if acc.accelerator_type == accelerator_type
    ]


def is_accelerator_available(accelerator_type: AcceleratorType) -> bool:
    """
    Check if a specific type of accelerator is available.

    Args:
        accelerator_type: The type of accelerator to check.

    Returns:
        True if at least one accelerator of the type is available.
    """
    return len(get_accelerator_by_type(accelerator_type)) > 0


def format_platform_info(info: PlatformInfo) -> str:
    """Format platform information as a human-readable string."""
    lines = [
        f"Platform: {info.os_name} {info.os_version}",
        f"Architecture: {info.architecture}",
        f"Python: {info.python_version}",
        f"Default Accelerator: {info.default_accelerator_type.value}",
        f"Accelerators ({len(info.accelerators)}):",
    ]

    for acc in info.accelerators:
        memory_gb = acc.memory_total_bytes / (1024**3)
        cc = f" (CC {acc.compute_capability})" if acc.compute_capability else ""
        lines.append(f"  [{acc.device_index}] {acc.device_name}{cc} - {memory_gb:.1f} GB")

    return "\n".join(lines)

#!/usr/bin/env python3
# ruff: noqa: T201
"""
Test script for NVIDIA GPU detection and monitoring.

Run this script on a system with NVIDIA GPUs to verify the detection
and monitoring functionality works correctly.

This is a standalone script, not a pytest test module.

Usage:
    python scripts/test_nvidia_detection.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def check_nvidia_available():
    """Check if NVIDIA GPUs are available."""
    print("=" * 60)
    print("Testing NVIDIA GPU Availability")
    print("=" * 60)

    from exo.worker.utils.nvidia_monitor import (
        get_nvidia_device_count,
        is_nvidia_available,
    )

    available = is_nvidia_available()
    print(f"NVIDIA available: {available}")

    if available:
        count = get_nvidia_device_count()
        print(f"Device count: {count}")
        return True
    else:
        print("No NVIDIA GPUs detected (pynvml not installed or no GPUs)")
        return False


def check_gpu_detection():
    """Check GPU detection and enumeration."""
    print("\n" + "=" * 60)
    print("Testing GPU Detection")
    print("=" * 60)

    from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus, format_gpu_info

    try:
        gpus = detect_nvidia_gpus()
        print(f"Detected {len(gpus)} GPU(s):\n")

        for gpu in gpus:
            print(format_gpu_info(gpu))
            print()

        return gpus
    except Exception as e:
        print(f"GPU detection failed: {e}")
        return []


async def check_gpu_metrics():
    """Check real-time GPU metrics collection."""
    print("\n" + "=" * 60)
    print("Testing GPU Metrics")
    print("=" * 60)

    from exo.worker.utils.nvidia_monitor import (
        format_gpu_metrics,
        get_nvidia_metrics_async,
    )

    try:
        metrics = await get_nvidia_metrics_async()
        print(f"Driver version: {metrics.driver_version}")
        print(f"NVML version: {metrics.nvml_version}")
        print(f"Timestamp: {metrics.timestamp}")
        print(f"Total GPUs: {len(metrics.gpus)}")
        print(f"Total memory used: {metrics.total_memory_used_bytes / (1024**3):.2f} GB")
        print(f"Total memory: {metrics.total_memory_total_bytes / (1024**3):.2f} GB")
        print(f"Average utilization: {metrics.average_utilization:.1f}%")
        print(f"Max temperature: {metrics.max_temperature:.0f}°C")
        print(f"Total power draw: {metrics.total_power_draw:.1f}W")
        print()

        for gpu_metric in metrics.gpus:
            print(format_gpu_metrics(gpu_metric))
            print()

        return metrics
    except Exception as e:
        print(f"GPU metrics collection failed: {e}")
        return None


def check_platform_detection():
    """Check unified platform detection."""
    print("\n" + "=" * 60)
    print("Testing Platform Detection")
    print("=" * 60)

    from exo.worker.utils.platform_detection import (
        detect_platform,
        format_platform_info,
    )

    platform_info = detect_platform()
    print(format_platform_info(platform_info))
    print()
    print(f"Has GPU: {platform_info.has_gpu}")
    print(f"Total GPU memory: {platform_info.total_gpu_memory_bytes / (1024**3):.2f} GB")

    return platform_info


async def check_gpu_profiles():
    """Check GPU performance profiles."""
    print("\n" + "=" * 60)
    print("Testing GPU Performance Profiles")
    print("=" * 60)

    from exo.worker.utils.profile import get_gpu_profiles

    profiles = await get_gpu_profiles()
    print(f"Found {len(profiles)} GPU profile(s):\n")

    for profile in profiles:
        print(f"GPU {profile.device_index}: {profile.device_name}")
        print(f"  Accelerator type: {profile.accelerator_type.value}")
        print(f"  Utilization: {profile.utilization_percent:.1f}%")
        print(f"  Memory: {profile.memory.used_gb:.2f} / {profile.memory.total_gb:.2f} GB")
        print(f"  Temperature: {profile.temperature_celsius:.0f}°C")
        print(f"  Power: {profile.power_watts:.1f} / {profile.power_limit_watts:.1f}W")
        print(f"  Clock: {profile.clock_mhz} MHz")
        print()

    return profiles


async def main():
    """Run all checks."""
    print("NVIDIA GPU Detection and Monitoring Test Suite")
    print("=" * 60)
    print()

    # Check 1: Check availability
    available = check_nvidia_available()
    if not available:
        print("\nNVIDIA GPUs not available. Exiting.")
        return 1

    # Check 2: Detect GPUs
    gpus = check_gpu_detection()
    if not gpus:
        print("\nNo GPUs detected. Exiting.")
        return 1

    # Check 3: Get metrics
    metrics = await check_gpu_metrics()
    if not metrics:
        print("\nMetrics collection failed.")
        return 1

    # Check 4: Platform detection
    check_platform_detection()

    # Check 5: GPU profiles
    await check_gpu_profiles()

    print("\n" + "=" * 60)
    print("All checks completed successfully!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

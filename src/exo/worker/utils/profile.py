import asyncio
import os
import platform
import sys
from typing import Any, Callable, Coroutine

import anyio
from loguru import logger

from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    AcceleratorType,
    GpuMemoryProfile,
    GpuPerformanceProfile,
    MemoryPerformanceProfile,
    NodePerformanceProfile,
    SystemPerformanceProfile,
)

from .macmon import (
    MacMonError,
    Metrics,
)
from .macmon import (
    get_metrics_async as macmon_get_metrics_async,
)
from .system_info import (
    get_friendly_name,
    get_model_and_chip,
    get_network_interfaces,
)


async def get_metrics_async() -> Metrics | None:
    """Return detailed Metrics on macOS or a minimal fallback elsewhere."""

    if platform.system().lower() == "darwin":
        return await macmon_get_metrics_async()


async def get_gpu_profiles() -> list[GpuPerformanceProfile]:
    """Get performance profiles for all available GPUs.

    This collects both static info (compute capability, NVLink support) and
    runtime metrics (utilization, memory usage, temperature) for all GPUs.
    """
    profiles: list[GpuPerformanceProfile] = []

    # Try NVIDIA GPUs (Linux/Windows)
    if sys.platform in ("linux", "win32"):
        try:
            from .nvidia_monitor import (
                NvidiaMonitorError,
                detect_nvidia_gpus,
                get_nvidia_metrics_async,
            )

            try:
                # Get static GPU info (compute capability, NVLink support, etc.)
                gpu_infos = detect_nvidia_gpus()
                gpu_info_by_index = {info.index: info for info in gpu_infos}

                # Get runtime metrics (utilization, memory, temperature, etc.)
                metrics = await get_nvidia_metrics_async()
                for gpu in metrics.gpus:
                    # Get static info for this GPU
                    gpu_info = gpu_info_by_index.get(gpu.index)

                    # Format compute capability as string (e.g., "8.0")
                    compute_cap = None
                    if gpu_info and gpu_info.compute_capability:
                        major, minor = gpu_info.compute_capability
                        compute_cap = f"{major}.{minor}"

                    profiles.append(
                        GpuPerformanceProfile(
                            device_index=gpu.index,
                            device_name=gpu.name,
                            accelerator_type=AcceleratorType.NVIDIA_CUDA,
                            utilization_percent=gpu.utilization_gpu,
                            memory=GpuMemoryProfile(
                                used_bytes=gpu.memory_used_bytes,
                                free_bytes=gpu.memory_free_bytes,
                                total_bytes=gpu.memory_total_bytes,
                            ),
                            temperature_celsius=gpu.temperature_c,
                            power_watts=gpu.power_draw_watts,
                            power_limit_watts=gpu.power_limit_watts,
                            clock_mhz=gpu.clock_graphics_mhz,
                            compute_capability=compute_cap,
                            nvlink_supported=gpu_info.nvlink_supported
                            if gpu_info
                            else False,
                            device_uuid=gpu_info.uuid if gpu_info else None,
                        )
                    )
            except NvidiaMonitorError:
                pass
        except ImportError:
            pass

    return profiles


def get_memory_profile() -> MemoryPerformanceProfile:
    """Construct a MemoryPerformanceProfile using psutil"""
    override_memory_env = os.getenv("OVERRIDE_MEMORY_MB")
    override_memory: int | None = (
        Memory.from_mb(int(override_memory_env)).in_bytes
        if override_memory_env
        else None
    )

    return MemoryPerformanceProfile.from_psutil(override_memory=override_memory)


async def start_polling_memory_metrics(
    callback: Callable[[MemoryPerformanceProfile], Coroutine[Any, Any, None]],
    *,
    poll_interval_s: float = 0.5,
) -> None:
    """Continuously poll and emit memory-only metrics at a faster cadence.

    Parameters
    - callback: coroutine called with a fresh MemoryPerformanceProfile each tick
    - poll_interval_s: interval between polls
    """
    while True:
        try:
            mem = get_memory_profile()
            await callback(mem)
        except MacMonError as e:
            logger.opt(exception=e).error("Memory Monitor encountered error")
        finally:
            await anyio.sleep(poll_interval_s)


async def start_polling_node_metrics(
    callback: Callable[[NodePerformanceProfile], Coroutine[Any, Any, None]],
):
    """Poll and emit full node performance metrics including GPU info.

    This function works on all platforms:
    - macOS: Includes Apple Silicon metrics (GPU usage, ANE power, etc.)
    - Linux/Windows: Includes NVIDIA GPU metrics if available
    """
    poll_interval_s = 1.0
    while True:
        try:
            # Get macOS-specific metrics (None on other platforms)
            metrics = await get_metrics_async()

            network_interfaces = get_network_interfaces()
            # these awaits could be joined but realistically they should be cached
            model_id, chip_id = await get_model_and_chip()
            friendly_name = await get_friendly_name()

            # do the memory profile last to get a fresh reading to not conflict with the other memory profiling loop
            memory_profile = get_memory_profile()

            # Get GPU profiles for all platforms (NVIDIA on Linux/Windows)
            gpu_profiles = await get_gpu_profiles()

            # Build system profile - use macOS metrics if available, defaults otherwise
            if metrics is not None:
                system_profile = SystemPerformanceProfile(
                    gpu_usage=metrics.gpu_usage[1],
                    temp=metrics.temp.gpu_temp_avg,
                    sys_power=metrics.sys_power,
                    pcpu_usage=metrics.pcpu_usage[1],
                    ecpu_usage=metrics.ecpu_usage[1],
                    ane_power=metrics.ane_power,
                )
            else:
                # Non-macOS platforms use defaults (GPU info is in gpu_profiles)
                system_profile = SystemPerformanceProfile()

            await callback(
                NodePerformanceProfile(
                    model_id=model_id,
                    chip_id=chip_id,
                    friendly_name=friendly_name,
                    network_interfaces=network_interfaces,
                    memory=memory_profile,
                    system=system_profile,
                    gpu_profiles=gpu_profiles,
                )
            )

        except asyncio.TimeoutError:
            logger.warning(
                "[resource_monitor] Operation timed out after 30s, skipping this cycle."
            )
        except MacMonError as e:
            logger.opt(exception=e).error("Resource Monitor encountered error")
            return
        finally:
            await anyio.sleep(poll_interval_s)

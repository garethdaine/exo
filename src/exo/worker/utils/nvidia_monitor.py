# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUnusedImport=false
# pyright: reportAttributeAccessIssue=false
"""
NVIDIA GPU monitoring using pynvml.

This module provides GPU detection, enumeration, and real-time metrics
monitoring for NVIDIA GPUs using the NVIDIA Management Library (NVML).

Features:
- GPU detection and enumeration
- Memory usage monitoring
- Temperature and power monitoring
- GPU utilization tracking
- Multi-GPU support

Usage:
    from exo.worker.utils.nvidia_monitor import (
        detect_nvidia_gpus,
        get_nvidia_metrics_async,
        NvidiaGpuInfo,
        NvidiaMetrics,
    )

    # Detect available GPUs
    gpus = detect_nvidia_gpus()
    for gpu in gpus:
        print(f"GPU {gpu.index}: {gpu.name} ({gpu.memory_total_mb} MB)")

    # Get real-time metrics
    metrics = await get_nvidia_metrics_async()
    for gpu_metric in metrics.gpus:
        print(f"GPU {gpu_metric.index}: {gpu_metric.utilization}% util, {gpu_metric.temperature_c}C")

Note:
    Type checking is relaxed in this module because pynvml is an optional
    dependency that may not be installed on all platforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass


class NvidiaMonitorError(Exception):
    """Exception raised for errors in NVIDIA monitoring functions."""


@dataclass(frozen=True)
class NvidiaGpuInfo:
    """Static information about an NVIDIA GPU."""

    index: int
    name: str
    uuid: str
    pci_bus_id: str
    memory_total_bytes: int
    memory_total_mb: int
    compute_capability: tuple[int, int]
    driver_version: str
    cuda_version: str | None
    is_multi_gpu_board: bool
    nvlink_supported: bool


@dataclass(frozen=True)
class NvidiaGpuMetrics:
    """Real-time metrics for an NVIDIA GPU."""

    index: int
    name: str
    utilization_gpu: float  # Percentage 0-100
    utilization_memory: float  # Percentage 0-100
    memory_used_bytes: int
    memory_free_bytes: int
    memory_total_bytes: int
    temperature_c: float
    power_draw_watts: float
    power_limit_watts: float
    clock_graphics_mhz: int
    clock_memory_mhz: int
    clock_sm_mhz: int
    fan_speed_percent: float | None  # None if not available
    pcie_throughput_tx_kbps: int | None
    pcie_throughput_rx_kbps: int | None


@dataclass(frozen=True)
class NvidiaMetrics:
    """Aggregated metrics for all NVIDIA GPUs."""

    gpus: list[NvidiaGpuMetrics]
    driver_version: str
    nvml_version: str
    timestamp: str

    @property
    def total_memory_used_bytes(self) -> int:
        """Total memory used across all GPUs."""
        return sum(gpu.memory_used_bytes for gpu in self.gpus)

    @property
    def total_memory_total_bytes(self) -> int:
        """Total memory across all GPUs."""
        return sum(gpu.memory_total_bytes for gpu in self.gpus)

    @property
    def average_utilization(self) -> float:
        """Average GPU utilization across all GPUs."""
        if not self.gpus:
            return 0.0
        return sum(gpu.utilization_gpu for gpu in self.gpus) / len(self.gpus)

    @property
    def max_temperature(self) -> float:
        """Maximum temperature across all GPUs."""
        if not self.gpus:
            return 0.0
        return max(gpu.temperature_c for gpu in self.gpus)

    @property
    def total_power_draw(self) -> float:
        """Total power draw across all GPUs in watts."""
        return sum(gpu.power_draw_watts for gpu in self.gpus)


def is_nvidia_available() -> bool:
    """
    Check if NVIDIA GPUs are available on this system.

    Returns:
        True if pynvml is installed and at least one NVIDIA GPU is detected.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return device_count > 0
    except (ImportError, Exception):
        return False


def get_nvidia_device_count() -> int:
    """
    Get the number of NVIDIA GPUs in the system.

    Returns:
        Number of NVIDIA GPUs, or 0 if pynvml is not available.
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        pynvml.nvmlShutdown()
        return count
    except (ImportError, Exception):
        return 0


def detect_nvidia_gpus() -> list[NvidiaGpuInfo]:
    """
    Detect and enumerate all NVIDIA GPUs in the system.

    Returns:
        List of NvidiaGpuInfo objects for each detected GPU.

    Raises:
        NvidiaMonitorError: If pynvml is not available or initialization fails.
    """
    try:
        import pynvml
    except ImportError as e:
        raise NvidiaMonitorError(
            "pynvml is not installed. Install with: pip install nvidia-ml-py3"
        ) from e

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise NvidiaMonitorError(f"Failed to initialize NVML: {e}") from e

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        driver_version = pynvml.nvmlSystemGetDriverVersion()

        # Try to get CUDA version
        cuda_version: str | None = None
        try:
            cuda_ver_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            # Convert to string format (e.g., 12001 -> "12.1")
            if isinstance(cuda_ver_raw, int):
                major = cuda_ver_raw // 1000
                minor = (cuda_ver_raw % 1000) // 10
                cuda_version = f"{major}.{minor}"
            elif isinstance(cuda_ver_raw, str):
                cuda_version = cuda_ver_raw
        except (AttributeError, pynvml.NVMLError):
            pass

        gpus: list[NvidiaGpuInfo] = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            uuid = pynvml.nvmlDeviceGetUUID(handle)
            if isinstance(uuid, bytes):
                uuid = uuid.decode("utf-8")

            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            pci_bus_id = pci_info.busId
            if isinstance(pci_bus_id, bytes):
                pci_bus_id = pci_bus_id.decode("utf-8")

            # Get memory info - some GPUs (e.g., GB10 on DGX Spark) use unified memory
            # and don't support per-GPU memory queries
            memory_total_bytes = 0
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_bytes = memory_info.total
            except pynvml.NVMLError:
                # Unified memory system - fall back to system memory
                try:
                    import psutil

                    memory_total_bytes = psutil.virtual_memory().total
                    logger.debug(
                        f"GPU {i} uses unified memory, using system RAM: "
                        f"{memory_total_bytes / (1024**3):.1f} GB"
                    )
                except ImportError:
                    logger.warning(f"GPU {i}: Cannot determine memory (psutil not available)")

            # Get compute capability - may not be available on all GPUs
            major, minor = 0, 0
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
            except (pynvml.NVMLError, AttributeError):
                # Some drivers don't expose this API
                logger.debug(f"GPU {i}: Compute capability not available")

            # Check for NVLink support
            nvlink_supported = False
            try:
                # NVLink is present if we can query link state for link 0
                if hasattr(pynvml, "nvmlDeviceGetNvLinkState"):
                    pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                    nvlink_supported = True
            except (pynvml.NVMLError, AttributeError):
                pass

            # Check for multi-GPU board
            is_multi_gpu = False
            try:
                is_multi_gpu = pynvml.nvmlDeviceGetMultiGpuBoard(handle) == 1
            except pynvml.NVMLError:
                pass

            gpus.append(
                NvidiaGpuInfo(
                    index=i,
                    name=name,
                    uuid=uuid,
                    pci_bus_id=pci_bus_id,
                    memory_total_bytes=memory_total_bytes,
                    memory_total_mb=memory_total_bytes // (1024 * 1024),
                    compute_capability=(major, minor),
                    driver_version=driver_version,
                    cuda_version=cuda_version,
                    is_multi_gpu_board=is_multi_gpu,
                    nvlink_supported=nvlink_supported,
                )
            )

        return gpus

    finally:
        pynvml.nvmlShutdown()


def get_nvidia_metrics() -> NvidiaMetrics:
    """
    Get real-time metrics for all NVIDIA GPUs.

    Returns:
        NvidiaMetrics object with current metrics for all GPUs.

    Raises:
        NvidiaMonitorError: If pynvml is not available or metrics collection fails.
    """
    import datetime

    try:
        import pynvml
    except ImportError as e:
        raise NvidiaMonitorError(
            "pynvml is not installed. Install with: pip install nvidia-ml-py3"
        ) from e

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as e:
        raise NvidiaMonitorError(f"Failed to initialize NVML: {e}") from e

    try:
        device_count = pynvml.nvmlDeviceGetCount()
        driver_version = pynvml.nvmlSystemGetDriverVersion()
        nvml_version = pynvml.nvmlSystemGetNVMLVersion()

        gpu_metrics: list[NvidiaGpuMetrics] = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            # Get utilization rates
            utilization_gpu = 0.0
            utilization_memory = 0.0
            try:
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization_gpu = float(utilization.gpu)
                utilization_memory = float(utilization.memory)
            except pynvml.NVMLError:
                pass

            # Get memory info - handle unified memory systems
            memory_used_bytes = 0
            memory_free_bytes = 0
            memory_total_bytes = 0
            try:
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used_bytes = memory_info.used
                memory_free_bytes = memory_info.free
                memory_total_bytes = memory_info.total
            except pynvml.NVMLError:
                # Unified memory system - use system memory stats
                try:
                    import psutil

                    vm = psutil.virtual_memory()
                    memory_total_bytes = vm.total
                    memory_used_bytes = vm.total - vm.available
                    memory_free_bytes = vm.available
                except ImportError:
                    pass

            # Get temperature
            temperature = 0.0
            try:
                temperature = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except pynvml.NVMLError:
                pass

            # Get power usage
            power_draw = 0.0
            power_limit = 0.0
            try:
                power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
                power_limit = (
                    pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
                )  # mW to W
            except pynvml.NVMLError:
                pass

            # Get clock speeds
            clock_graphics = 0
            clock_memory = 0
            clock_sm = 0
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                clock_memory = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
                clock_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            except pynvml.NVMLError:
                pass

            # Get fan speed (may not be available on all GPUs)
            fan_speed: float | None = None
            try:
                fan_speed = float(pynvml.nvmlDeviceGetFanSpeed(handle))
            except pynvml.NVMLError:
                pass

            # Get PCIe throughput
            pcie_tx: int | None = None
            pcie_rx: int | None = None
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
            except pynvml.NVMLError:
                pass

            gpu_metrics.append(
                NvidiaGpuMetrics(
                    index=i,
                    name=name,
                    utilization_gpu=utilization_gpu,
                    utilization_memory=utilization_memory,
                    memory_used_bytes=memory_used_bytes,
                    memory_free_bytes=memory_free_bytes,
                    memory_total_bytes=memory_total_bytes,
                    temperature_c=temperature,
                    power_draw_watts=power_draw,
                    power_limit_watts=power_limit,
                    clock_graphics_mhz=clock_graphics,
                    clock_memory_mhz=clock_memory,
                    clock_sm_mhz=clock_sm,
                    fan_speed_percent=fan_speed,
                    pcie_throughput_tx_kbps=pcie_tx,
                    pcie_throughput_rx_kbps=pcie_rx,
                )
            )

        return NvidiaMetrics(
            gpus=gpu_metrics,
            driver_version=driver_version,
            nvml_version=nvml_version,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        )

    finally:
        pynvml.nvmlShutdown()


async def get_nvidia_metrics_async() -> NvidiaMetrics:
    """
    Asynchronously get real-time metrics for all NVIDIA GPUs.

    This is a thin async wrapper around get_nvidia_metrics() that runs
    the synchronous call in a thread pool to avoid blocking the event loop.

    Returns:
        NvidiaMetrics object with current metrics for all GPUs.

    Raises:
        NvidiaMonitorError: If pynvml is not available or metrics collection fails.
    """
    import anyio

    return await anyio.to_thread.run_sync(get_nvidia_metrics)


def get_gpu_memory_info(device_index: int = 0) -> tuple[int, int, int]:
    """
    Get memory information for a specific GPU.

    Args:
        device_index: The GPU index to query.

    Returns:
        Tuple of (used_bytes, free_bytes, total_bytes).

    Raises:
        NvidiaMonitorError: If the GPU cannot be queried.
    """
    try:
        import pynvml
    except ImportError as e:
        raise NvidiaMonitorError(
            "pynvml is not installed. Install with: pip install nvidia-ml-py3"
        ) from e

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return (memory_info.used, memory_info.free, memory_info.total)
    except pynvml.NVMLError as e:
        raise NvidiaMonitorError(f"Failed to get memory info for GPU {device_index}: {e}") from e
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def format_gpu_info(gpu: NvidiaGpuInfo) -> str:
    """Format GPU info as a human-readable string."""
    cc = f"{gpu.compute_capability[0]}.{gpu.compute_capability[1]}"
    memory_gb = gpu.memory_total_mb / 1024
    return (
        f"GPU {gpu.index}: {gpu.name}\n"
        f"  UUID: {gpu.uuid}\n"
        f"  Memory: {memory_gb:.1f} GB\n"
        f"  Compute Capability: {cc}\n"
        f"  Driver: {gpu.driver_version}\n"
        f"  CUDA: {gpu.cuda_version or 'N/A'}\n"
        f"  NVLink: {'Yes' if gpu.nvlink_supported else 'No'}"
    )


def format_gpu_metrics(metrics: NvidiaGpuMetrics) -> str:
    """Format GPU metrics as a human-readable string."""
    memory_used_gb = metrics.memory_used_bytes / (1024**3)
    memory_total_gb = metrics.memory_total_bytes / (1024**3)
    return (
        f"GPU {metrics.index}: {metrics.name}\n"
        f"  Utilization: {metrics.utilization_gpu:.1f}% GPU, {metrics.utilization_memory:.1f}% Memory\n"
        f"  Memory: {memory_used_gb:.1f} / {memory_total_gb:.1f} GB\n"
        f"  Temperature: {metrics.temperature_c:.0f}C\n"
        f"  Power: {metrics.power_draw_watts:.1f} / {metrics.power_limit_watts:.1f} W\n"
        f"  Clocks: {metrics.clock_graphics_mhz} MHz (Graphics), {metrics.clock_memory_mhz} MHz (Memory)"
    )

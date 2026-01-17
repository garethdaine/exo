from enum import Enum
from typing import Self

import psutil

from exo.shared.types.memory import Memory
from exo.utils.pydantic_ext import CamelCaseModel


class AcceleratorType(str, Enum):
    """Type of hardware accelerator."""

    APPLE_SILICON = "apple_silicon"
    NVIDIA_CUDA = "nvidia_cuda"
    AMD_ROCM = "amd_rocm"
    INTEL_ONEAPI = "intel_oneapi"
    VULKAN = "vulkan"
    CPU_ONLY = "cpu_only"


class GpuMemoryProfile(CamelCaseModel):
    """Memory profile for a GPU device."""

    used_bytes: int
    free_bytes: int
    total_bytes: int

    @property
    def used_gb(self) -> float:
        """Memory used in gigabytes."""
        return self.used_bytes / (1024**3)

    @property
    def total_gb(self) -> float:
        """Total memory in gigabytes."""
        return self.total_bytes / (1024**3)

    @property
    def utilization_percent(self) -> float:
        """Memory utilization as a percentage."""
        if self.total_bytes == 0:
            return 0.0
        return (self.used_bytes / self.total_bytes) * 100


class GpuPerformanceProfile(CamelCaseModel):
    """Performance profile for a single GPU device."""

    device_index: int
    device_name: str
    accelerator_type: AcceleratorType
    utilization_percent: float = 0.0
    memory: GpuMemoryProfile
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    power_limit_watts: float = 0.0
    clock_mhz: int = 0
    # Compute capability for NVIDIA GPUs (e.g., "8.0" for SM 8.0)
    # Used for placement decisions to ensure GPU meets model requirements
    compute_capability: str | None = None
    # Whether this GPU supports NVLink for high-bandwidth multi-GPU communication
    nvlink_supported: bool = False
    # Device UUID for unique identification
    device_uuid: str | None = None


class MemoryPerformanceProfile(CamelCaseModel):
    ram_total: Memory
    ram_available: Memory
    swap_total: Memory
    swap_available: Memory

    @classmethod
    def from_bytes(
        cls, *, ram_total: int, ram_available: int, swap_total: int, swap_available: int
    ) -> Self:
        return cls(
            ram_total=Memory.from_bytes(ram_total),
            ram_available=Memory.from_bytes(ram_available),
            swap_total=Memory.from_bytes(swap_total),
            swap_available=Memory.from_bytes(swap_available),
        )

    @classmethod
    def from_psutil(cls, *, override_memory: int | None) -> Self:
        vm = psutil.virtual_memory()
        sm = psutil.swap_memory()

        return cls.from_bytes(
            ram_total=vm.total,
            ram_available=vm.available if override_memory is None else override_memory,
            swap_total=sm.total,
            swap_available=sm.free,
        )


class SystemPerformanceProfile(CamelCaseModel):
    # TODO: flops_fp16: float

    gpu_usage: float = 0.0
    temp: float = 0.0
    sys_power: float = 0.0
    pcpu_usage: float = 0.0
    ecpu_usage: float = 0.0
    ane_power: float = 0.0


class NetworkInterfaceInfo(CamelCaseModel):
    name: str
    ip_address: str


class NodePerformanceProfile(CamelCaseModel):
    model_id: str
    chip_id: str
    friendly_name: str
    memory: MemoryPerformanceProfile
    network_interfaces: list[NetworkInterfaceInfo] = []
    system: SystemPerformanceProfile
    # GPU profiles for all accelerators on this node (NVIDIA, etc.)
    # Used by Master for GPU-aware placement decisions
    gpu_profiles: list[GpuPerformanceProfile] = []

    @property
    def total_gpu_memory_bytes(self) -> int:
        """Total GPU memory across all GPUs on this node."""
        return sum(gpu.memory.total_bytes for gpu in self.gpu_profiles)

    @property
    def available_gpu_memory_bytes(self) -> int:
        """Available GPU memory across all GPUs on this node."""
        return sum(gpu.memory.free_bytes for gpu in self.gpu_profiles)

    @property
    def has_cuda_gpus(self) -> bool:
        """Whether this node has NVIDIA CUDA GPUs."""
        return any(
            gpu.accelerator_type == AcceleratorType.NVIDIA_CUDA
            for gpu in self.gpu_profiles
        )


class ConnectionProfile(CamelCaseModel):
    throughput: float
    latency: float
    jitter: float

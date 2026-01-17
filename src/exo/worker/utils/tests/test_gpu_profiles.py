# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""Tests for GPU profile integration.

These tests verify that GPU profiling data flows correctly from the
worker utilities to the node performance profiles.
"""

import sys

import pytest

# Skip NVIDIA-specific tests on macOS
pytestmark = []


class TestGpuProfileIntegration:
    """Tests for GPU profile integration with NodePerformanceProfile."""

    async def test_get_gpu_profiles_returns_list(self) -> None:
        """Test that get_gpu_profiles returns a list."""
        from exo.worker.utils.profile import get_gpu_profiles

        profiles = await get_gpu_profiles()
        assert isinstance(profiles, list)

    @pytest.mark.skipif(sys.platform == "darwin", reason="Test for NVIDIA systems")
    async def test_gpu_profiles_on_nvidia_system(self) -> None:
        """Test GPU profiles on systems with NVIDIA GPUs."""
        try:
            from exo.worker.utils.nvidia_monitor import is_nvidia_available
        except ImportError:
            pytest.skip("pynvml not installed")

        if not is_nvidia_available():
            pytest.skip("No NVIDIA GPUs available")

        from exo.shared.types.profiling import AcceleratorType
        from exo.worker.utils.profile import get_gpu_profiles

        profiles = await get_gpu_profiles()
        assert len(profiles) > 0

        # Verify profile structure
        profile = profiles[0]
        assert profile.accelerator_type == AcceleratorType.NVIDIA_CUDA
        assert profile.device_index >= 0
        assert profile.device_name  # Non-empty string
        assert profile.memory is not None
        assert profile.memory.total_bytes > 0

    @pytest.mark.skipif(sys.platform != "darwin", reason="Test for macOS only")
    async def test_gpu_profiles_on_macos(self) -> None:
        """Test GPU profiles on macOS (should return empty list for discrete GPUs)."""
        from exo.worker.utils.profile import get_gpu_profiles

        profiles = await get_gpu_profiles()
        # On macOS, we don't return Apple Silicon as "GPU profiles"
        # (Apple Silicon is handled differently via macmon)
        assert isinstance(profiles, list)


class TestNodePerformanceProfileGpu:
    """Tests for GPU-related properties of NodePerformanceProfile."""

    def test_node_profile_gpu_properties_no_gpus(self) -> None:
        """Test GPU properties when no GPUs are present."""
        from exo.shared.types.profiling import (
            MemoryPerformanceProfile,
            NodePerformanceProfile,
            SystemPerformanceProfile,
        )

        profile = NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="Test Node",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=16 * 1024**3,
                ram_available=8 * 1024**3,
                swap_total=4 * 1024**3,
                swap_available=4 * 1024**3,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
            gpu_profiles=[],  # No GPUs
        )

        assert profile.total_gpu_memory_bytes == 0
        assert profile.available_gpu_memory_bytes == 0
        assert profile.has_cuda_gpus is False

    def test_node_profile_gpu_properties_with_gpus(self) -> None:
        """Test GPU properties when GPUs are present."""
        from exo.shared.types.profiling import (
            AcceleratorType,
            GpuMemoryProfile,
            GpuPerformanceProfile,
            MemoryPerformanceProfile,
            NodePerformanceProfile,
            SystemPerformanceProfile,
        )

        gpu_profile = GpuPerformanceProfile(
            device_index=0,
            device_name="NVIDIA Test GPU",
            accelerator_type=AcceleratorType.NVIDIA_CUDA,
            utilization_percent=50.0,
            memory=GpuMemoryProfile(
                used_bytes=4 * 1024**3,
                free_bytes=12 * 1024**3,
                total_bytes=16 * 1024**3,
            ),
            temperature_celsius=65.0,
            power_watts=150.0,
            power_limit_watts=350.0,
            clock_mhz=1800,
            compute_capability="8.0",
            nvlink_supported=True,
            device_uuid="GPU-12345",
        )

        profile = NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="Test Node",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=64 * 1024**3,
                ram_available=32 * 1024**3,
                swap_total=8 * 1024**3,
                swap_available=8 * 1024**3,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
            gpu_profiles=[gpu_profile],
        )

        assert profile.total_gpu_memory_bytes == 16 * 1024**3
        assert profile.available_gpu_memory_bytes == 12 * 1024**3
        assert profile.has_cuda_gpus is True

    def test_node_profile_multiple_gpus(self) -> None:
        """Test GPU properties with multiple GPUs."""
        from exo.shared.types.profiling import (
            AcceleratorType,
            GpuMemoryProfile,
            GpuPerformanceProfile,
            MemoryPerformanceProfile,
            NodePerformanceProfile,
            SystemPerformanceProfile,
        )

        gpu_profiles = [
            GpuPerformanceProfile(
                device_index=i,
                device_name=f"NVIDIA Test GPU {i}",
                accelerator_type=AcceleratorType.NVIDIA_CUDA,
                utilization_percent=50.0,
                memory=GpuMemoryProfile(
                    used_bytes=4 * 1024**3,
                    free_bytes=12 * 1024**3,
                    total_bytes=16 * 1024**3,
                ),
                temperature_celsius=65.0,
                power_watts=150.0,
                power_limit_watts=350.0,
                clock_mhz=1800,
                compute_capability="8.0",
                nvlink_supported=True,
                device_uuid=f"GPU-{i}",
            )
            for i in range(4)  # 4 GPUs
        ]

        profile = NodePerformanceProfile(
            model_id="test",
            chip_id="test",
            friendly_name="Test Node",
            memory=MemoryPerformanceProfile.from_bytes(
                ram_total=256 * 1024**3,
                ram_available=128 * 1024**3,
                swap_total=32 * 1024**3,
                swap_available=32 * 1024**3,
            ),
            network_interfaces=[],
            system=SystemPerformanceProfile(),
            gpu_profiles=gpu_profiles,
        )

        assert profile.total_gpu_memory_bytes == 4 * 16 * 1024**3  # 64 GB total
        assert profile.available_gpu_memory_bytes == 4 * 12 * 1024**3  # 48 GB free
        assert profile.has_cuda_gpus is True

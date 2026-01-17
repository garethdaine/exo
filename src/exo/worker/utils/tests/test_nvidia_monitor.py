# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAny=false
"""Tests for NVIDIA GPU monitoring.

These tests require NVIDIA hardware and will be skipped on macOS or systems
without NVIDIA GPUs.
"""

import sys

import pytest

# Skip all tests in this module on macOS
pytestmark = [
    pytest.mark.skipif(sys.platform == "darwin", reason="NVIDIA tests not applicable on macOS"),
]


def is_nvidia_available() -> bool:
    """Check if NVIDIA testing is available."""
    try:
        from exo.worker.utils.nvidia_monitor import is_nvidia_available as check_nvidia

        return check_nvidia()
    except ImportError:
        return False


class TestNvidiaAvailability:
    """Tests for NVIDIA availability detection."""

    def test_is_nvidia_available_returns_bool(self) -> None:
        """Test that is_nvidia_available returns a boolean."""
        try:
            from exo.worker.utils.nvidia_monitor import (
                is_nvidia_available as check_nvidia,
            )

            result = check_nvidia()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("pynvml not installed")

    def test_get_nvidia_device_count_returns_int(self) -> None:
        """Test that get_nvidia_device_count returns an integer."""
        try:
            from exo.worker.utils.nvidia_monitor import get_nvidia_device_count

            count = get_nvidia_device_count()
            assert isinstance(count, int)
            assert count >= 0
        except ImportError:
            pytest.skip("pynvml not installed")


@pytest.mark.skipif(not is_nvidia_available(), reason="NVIDIA GPUs not available")
class TestNvidiaGpuDetection:
    """Tests for NVIDIA GPU detection."""

    def test_detect_nvidia_gpus_returns_list(self) -> None:
        """Test that detect_nvidia_gpus returns a list."""
        from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus

        gpus = detect_nvidia_gpus()
        assert isinstance(gpus, list)
        assert len(gpus) > 0

    def test_gpu_info_has_required_fields(self) -> None:
        """Test that GPU info has all required fields."""
        from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus

        gpus = detect_nvidia_gpus()
        gpu = gpus[0]

        # Verify required fields
        assert hasattr(gpu, "index")
        assert hasattr(gpu, "name")
        assert hasattr(gpu, "uuid")
        assert hasattr(gpu, "memory_total_bytes")
        assert hasattr(gpu, "compute_capability")
        assert hasattr(gpu, "nvlink_supported")

    def test_gpu_info_values_are_valid(self) -> None:
        """Test that GPU info values are valid."""
        from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus

        gpus = detect_nvidia_gpus()
        gpu = gpus[0]

        assert gpu.index >= 0
        assert gpu.name  # Non-empty string
        assert gpu.uuid  # Non-empty string
        assert gpu.memory_total_bytes >= 0
        assert isinstance(gpu.compute_capability, tuple)
        assert len(gpu.compute_capability) == 2
        assert isinstance(gpu.nvlink_supported, bool)


@pytest.mark.skipif(not is_nvidia_available(), reason="NVIDIA GPUs not available")
class TestNvidiaMetrics:
    """Tests for NVIDIA GPU metrics collection."""

    def test_get_nvidia_metrics_returns_metrics(self) -> None:
        """Test that get_nvidia_metrics returns NvidiaMetrics."""
        from exo.worker.utils.nvidia_monitor import NvidiaMetrics, get_nvidia_metrics

        metrics = get_nvidia_metrics()
        assert isinstance(metrics, NvidiaMetrics)

    def test_metrics_has_gpu_data(self) -> None:
        """Test that metrics contains GPU data."""
        from exo.worker.utils.nvidia_monitor import get_nvidia_metrics

        metrics = get_nvidia_metrics()
        assert len(metrics.gpus) > 0

    def test_gpu_metrics_has_required_fields(self) -> None:
        """Test that GPU metrics has all required fields."""
        from exo.worker.utils.nvidia_monitor import get_nvidia_metrics

        metrics = get_nvidia_metrics()
        gpu = metrics.gpus[0]

        # Verify required fields
        assert hasattr(gpu, "index")
        assert hasattr(gpu, "name")
        assert hasattr(gpu, "utilization_gpu")
        assert hasattr(gpu, "memory_used_bytes")
        assert hasattr(gpu, "memory_free_bytes")
        assert hasattr(gpu, "memory_total_bytes")
        assert hasattr(gpu, "temperature_c")
        assert hasattr(gpu, "power_draw_watts")

    def test_gpu_metrics_values_are_valid(self) -> None:
        """Test that GPU metrics values are within valid ranges."""
        from exo.worker.utils.nvidia_monitor import get_nvidia_metrics

        metrics = get_nvidia_metrics()
        gpu = metrics.gpus[0]

        assert gpu.index >= 0
        assert 0 <= gpu.utilization_gpu <= 100
        assert gpu.memory_used_bytes >= 0
        assert gpu.memory_free_bytes >= 0
        assert gpu.memory_total_bytes > 0
        assert gpu.temperature_c >= 0  # Temperature should be non-negative

    def test_metrics_aggregate_properties(self) -> None:
        """Test that metrics aggregate properties work correctly."""
        from exo.worker.utils.nvidia_monitor import get_nvidia_metrics

        metrics = get_nvidia_metrics()

        # Test aggregate properties
        assert metrics.total_memory_used_bytes >= 0
        assert metrics.total_memory_total_bytes > 0
        assert 0 <= metrics.average_utilization <= 100
        assert metrics.max_temperature >= 0
        assert metrics.total_power_draw >= 0


@pytest.mark.skipif(not is_nvidia_available(), reason="NVIDIA GPUs not available")
class TestNvidiaMetricsAsync:
    """Tests for async NVIDIA metrics collection."""

    async def test_get_nvidia_metrics_async_returns_metrics(self) -> None:
        """Test that async metrics retrieval works."""
        from exo.worker.utils.nvidia_monitor import (
            NvidiaMetrics,
            get_nvidia_metrics_async,
        )

        metrics = await get_nvidia_metrics_async()
        assert isinstance(metrics, NvidiaMetrics)
        assert len(metrics.gpus) > 0


@pytest.mark.skipif(not is_nvidia_available(), reason="NVIDIA GPUs not available")
class TestNvidiaFormatters:
    """Tests for NVIDIA output formatters."""

    def test_format_gpu_info(self) -> None:
        """Test that GPU info formatting works."""
        from exo.worker.utils.nvidia_monitor import detect_nvidia_gpus, format_gpu_info

        gpus = detect_nvidia_gpus()
        formatted = format_gpu_info(gpus[0])

        assert isinstance(formatted, str)
        assert "GPU" in formatted
        assert gpus[0].name in formatted

    def test_format_gpu_metrics(self) -> None:
        """Test that GPU metrics formatting works."""
        from exo.worker.utils.nvidia_monitor import (
            format_gpu_metrics,
            get_nvidia_metrics,
        )

        metrics = get_nvidia_metrics()
        formatted = format_gpu_metrics(metrics.gpus[0])

        assert isinstance(formatted, str)
        assert "GPU" in formatted
        assert "Utilization" in formatted

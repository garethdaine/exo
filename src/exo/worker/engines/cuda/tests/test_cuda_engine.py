# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
# pyright: reportAny=false
"""Tests for CudaEngine implementation.

These tests require CUDA hardware and will be skipped on macOS or systems
without NVIDIA GPUs.
"""

import sys

import pytest

# Skip all tests in this module on macOS or if CUDA is not available
pytestmark = [
    pytest.mark.skipif(sys.platform == "darwin", reason="CUDA tests not applicable on macOS"),
]


def is_cuda_test_available() -> bool:
    """Check if CUDA testing is available."""
    try:
        from exo.worker.engines.cuda import is_cuda_available

        return is_cuda_available()
    except ImportError:
        return False


class TestCudaAvailability:
    """Tests for CUDA availability detection."""

    def test_is_cuda_available_returns_bool(self) -> None:
        """Test that is_cuda_available returns a boolean."""
        try:
            from exo.worker.engines.cuda import is_cuda_available

            result = is_cuda_available()
            assert isinstance(result, bool)
        except ImportError:
            pytest.skip("PyTorch not installed")


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestCudaEngine:
    """Tests for CudaEngine class."""

    def test_engine_instantiation(self) -> None:
        """Test that CudaEngine can be instantiated."""
        from exo.worker.engines.cuda import get_cuda_engine

        engine = get_cuda_engine()
        assert engine is not None

    def test_engine_implements_protocol(self) -> None:
        """Test that CudaEngine implements InferenceEngine protocol."""
        from exo.worker.engines.base import InferenceEngine
        from exo.worker.engines.cuda import get_cuda_engine

        engine = get_cuda_engine()
        assert isinstance(engine, InferenceEngine)

    def test_engine_has_required_methods(self) -> None:
        """Test that CudaEngine has all required methods."""
        from exo.worker.engines.cuda import get_cuda_engine

        engine = get_cuda_engine()
        required_methods = [
            "initialize_distributed",
            "load_model",
            "warmup",
            "generate",
            "cleanup",
            "get_info",
        ]
        for method in required_methods:
            assert hasattr(engine, method), f"CudaEngine missing method: {method}"

    def test_engine_get_info(self) -> None:
        """Test that CudaEngine.get_info returns correct information."""
        from exo.worker.engines.cuda import get_cuda_engine

        engine = get_cuda_engine()
        info = engine.get_info()

        assert info.name == "cuda"
        assert info.capabilities is not None
        assert info.capabilities.supports_tensor_parallelism is True
        assert info.capabilities.supports_pipeline_parallelism is True
        assert info.capabilities.supports_streaming is True


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestCudaDistributed:
    """Tests for CUDA distributed group classes."""

    def test_nccl_group_class_exists(self) -> None:
        """Test that NcclDistributedGroup class exists."""
        from exo.worker.engines.cuda.distributed import NcclDistributedGroup

        assert NcclDistributedGroup is not None

    def test_gloo_group_class_exists(self) -> None:
        """Test that GlooDistributedGroup class exists."""
        from exo.worker.engines.cuda.distributed import GlooDistributedGroup

        assert GlooDistributedGroup is not None

    def test_distributed_groups_implement_protocol(self) -> None:
        """Test that distributed groups implement DistributedGroup protocol."""
        from exo.worker.engines.cuda.distributed import (
            GlooDistributedGroup,
            NcclDistributedGroup,
        )

        # Check that they have the required methods (can't instantiate without real instances)
        for cls in [NcclDistributedGroup, GlooDistributedGroup]:
            assert hasattr(cls, "rank")
            assert hasattr(cls, "size")
            assert hasattr(cls, "cleanup")


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestCudaUtils:
    """Tests for CUDA utility functions."""

    def test_get_cuda_device_info(self) -> None:
        """Test getting CUDA device information."""
        from exo.worker.engines.cuda.utils_cuda import get_cuda_device_info

        info = get_cuda_device_info(0)
        assert info is not None
        assert info.name  # Should have a name
        assert info.total_memory_bytes > 0
        assert info.total_memory_gb > 0

    def test_get_optimal_dtype(self) -> None:
        """Test getting optimal dtype for CUDA device."""
        import torch

        from exo.worker.engines.cuda.utils_cuda import get_optimal_dtype

        dtype = get_optimal_dtype()
        assert dtype in [torch.float16, torch.bfloat16, torch.float32]

    def test_initialize_cuda_device(self) -> None:
        """Test initializing CUDA device."""

        from exo.worker.engines.cuda.utils_cuda import initialize_cuda_device

        device = initialize_cuda_device(0)
        assert device is not None
        assert "cuda" in str(device)

    def test_clear_cuda_cache(self) -> None:
        """Test clearing CUDA cache."""
        from exo.worker.engines.cuda.utils_cuda import clear_cuda_cache

        # Should not raise
        clear_cuda_cache()

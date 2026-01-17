# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
# pyright: reportAny=false
"""Tests for CUDA distributed communication groups.

These tests verify the NCCL and Gloo distributed group implementations.
Full distributed tests require multiple processes and CUDA hardware.
"""

import sys

import pytest

# Skip all tests in this module on macOS
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


class TestDistributedGroupImports:
    """Tests for distributed group imports."""

    def test_nccl_group_importable(self) -> None:
        """Test that NcclDistributedGroup can be imported."""
        try:
            from exo.worker.engines.cuda.distributed import NcclDistributedGroup

            assert NcclDistributedGroup is not None
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_gloo_group_importable(self) -> None:
        """Test that GlooDistributedGroup can be imported."""
        try:
            from exo.worker.engines.cuda.distributed import GlooDistributedGroup

            assert GlooDistributedGroup is not None
        except ImportError:
            pytest.skip("PyTorch not installed")


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestNcclDistributedGroup:
    """Tests for NcclDistributedGroup class."""

    def test_nccl_group_requires_cuda_nccl_instance(self) -> None:
        """Test that NcclDistributedGroup requires CudaNcclInstance."""
        from exo.shared.types.common import NodeId
        from exo.shared.types.memory import Memory
        from exo.shared.types.models import ModelId, ModelMetadata
        from exo.shared.types.worker.instances import (
            BoundInstance,
            CudaGlooInstance,
            InstanceId,
        )
        from exo.shared.types.worker.runners import RunnerId, ShardAssignments
        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.cuda.distributed import NcclDistributedGroup

        # Create a Gloo instance (wrong type)
        model_meta = ModelMetadata(
            model_id=ModelId("test/model"),
            n_layers=32,
            hidden_size=4096,
            storage_size=Memory.from_gb(8),
            supports_tensor=False,
        )

        runner_id = RunnerId()
        node_id = NodeId()
        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=32,
            n_layers=32,
        )

        shard_assignments = ShardAssignments(
            model_id=model_meta.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        )

        gloo_instance = CudaGlooInstance(
            instance_id=InstanceId(),
            shard_assignments=shard_assignments,
            master_addr="127.0.0.1",
            master_port=29500,
            device_ids=[0],
        )

        bound = BoundInstance(
            instance=gloo_instance,
            bound_shard=shard,
            runner_id=runner_id,
            node_id=node_id,
        )

        with pytest.raises(TypeError, match="CudaNcclInstance"):
            NcclDistributedGroup(bound)


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestGlooDistributedGroup:
    """Tests for GlooDistributedGroup class."""

    def test_gloo_group_requires_cuda_gloo_instance(self) -> None:
        """Test that GlooDistributedGroup requires CudaGlooInstance."""
        from exo.shared.types.common import NodeId
        from exo.shared.types.memory import Memory
        from exo.shared.types.models import ModelId, ModelMetadata
        from exo.shared.types.worker.instances import (
            BoundInstance,
            CudaNcclInstance,
            InstanceId,
        )
        from exo.shared.types.worker.runners import RunnerId, ShardAssignments
        from exo.shared.types.worker.shards import PipelineShardMetadata
        from exo.worker.engines.cuda.distributed import GlooDistributedGroup

        # Create an NCCL instance (wrong type)
        model_meta = ModelMetadata(
            model_id=ModelId("test/model"),
            n_layers=32,
            hidden_size=4096,
            storage_size=Memory.from_gb(8),
            supports_tensor=False,
        )

        runner_id = RunnerId()
        node_id = NodeId()
        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=0,
            world_size=1,
            start_layer=0,
            end_layer=32,
            n_layers=32,
        )

        shard_assignments = ShardAssignments(
            model_id=model_meta.model_id,
            runner_to_shard={runner_id: shard},
            node_to_runner={node_id: runner_id},
        )

        nccl_instance = CudaNcclInstance(
            instance_id=InstanceId(),
            shard_assignments=shard_assignments,
            nccl_unique_id="a" * 64,
            master_addr="127.0.0.1",
            master_port=29500,
            device_ids=[0],
        )

        bound = BoundInstance(
            instance=nccl_instance,
            bound_shard=shard,
            runner_id=runner_id,
            node_id=node_id,
        )

        with pytest.raises(TypeError, match="CudaGlooInstance"):
            GlooDistributedGroup(bound)


@pytest.mark.skipif(not is_cuda_test_available(), reason="CUDA not available")
class TestDistributedGroupProtocol:
    """Tests for DistributedGroup protocol compliance."""

    def test_nccl_group_has_protocol_methods(self) -> None:
        """Test that NcclDistributedGroup has required protocol methods."""
        from exo.worker.engines.cuda.distributed import NcclDistributedGroup

        assert hasattr(NcclDistributedGroup, "rank")
        assert hasattr(NcclDistributedGroup, "size")
        assert hasattr(NcclDistributedGroup, "cleanup")

    def test_gloo_group_has_protocol_methods(self) -> None:
        """Test that GlooDistributedGroup has required protocol methods."""
        from exo.worker.engines.cuda.distributed import GlooDistributedGroup

        assert hasattr(GlooDistributedGroup, "rank")
        assert hasattr(GlooDistributedGroup, "size")
        assert hasattr(GlooDistributedGroup, "cleanup")

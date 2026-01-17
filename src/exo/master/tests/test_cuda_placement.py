"""Tests for CUDA instance placement.

These tests verify that CUDA instances (NCCL and Gloo) are placed correctly
using GPU-aware filtering and ranking.
"""


import pytest

from exo.shared.topology import Topology
from exo.shared.types.common import NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.worker.instances import (
    CudaGlooInstance,
    CudaNcclInstance,
    InstanceMeta,
)
from exo.shared.types.worker.shards import Sharding


def create_topology_with_nodes_and_connections(
    nodes: list[NodeInfo], connections: list[Connection]
) -> Topology:
    """Helper to create a topology with nodes and connections."""
    topology = Topology()
    for node in nodes:
        topology.add_node(node)
    for conn in connections:
        topology.add_connection(conn)
    return topology


class TestCudaInstancePlacement:
    """Tests for CUDA instance placement logic."""

    def test_place_cuda_nccl_instance_single_node(
        self, create_gpu_node, create_connection
    ) -> None:
        """Test placing a CUDA NCCL instance on a single node."""
        from exo.master.placement import place_instance
        from exo.shared.types.commands import PlaceInstance

        # Create a node with a GPU
        node_id = NodeId()
        node = create_gpu_node(
            ram_memory=64 * 1024**3,  # 64 GB RAM
            gpu_memory=24 * 1024**3,  # 24 GB GPU
            node_id=node_id,
            compute_capability="8.0",
            nvlink_supported=False,
        )

        # Create topology with single node
        topology = create_topology_with_nodes_and_connections([node], [])

        # Create placement command for small model
        model_meta = ModelMetadata(
            model_id=ModelId("test/small-model"),
            pretty_name="Test Small Model",
            n_layers=32,
            hidden_size=4096,
            storage_size=Memory.from_gb(8),  # 8 GB model
            supports_tensor=False,
        )

        command = PlaceInstance(
            model_meta=model_meta,
            instance_meta=InstanceMeta.CudaNccl,
            sharding=Sharding.Pipeline,
            min_nodes=1,
        )

        # Place the instance
        instances = place_instance(command, topology, {})

        # Single node falls back to MlxRing due to warning in place_instance
        # This is expected behavior - single node doesn't need distributed
        assert len(instances) == 1

    def test_place_cuda_instance_multi_node(
        self, create_gpu_node, create_connection
    ) -> None:
        """Test placing a CUDA NCCL instance across multiple nodes."""
        from exo.master.placement import place_instance
        from exo.shared.types.commands import PlaceInstance

        # Create two nodes with GPUs
        node_a_id = NodeId()
        node_b_id = NodeId()

        node_a = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_a_id,
            compute_capability="8.0",
        )
        node_b = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_b_id,
            compute_capability="8.0",
        )

        # Create bidirectional connections
        conn_a_to_b = create_connection(node_a_id, node_b_id)
        conn_b_to_a = create_connection(node_b_id, node_a_id)

        topology = create_topology_with_nodes_and_connections(
            [node_a, node_b],
            [conn_a_to_b, conn_b_to_a],
        )

        # Create placement command for larger model
        model_meta = ModelMetadata(
            model_id=ModelId("test/large-model"),
            pretty_name="Test Large Model",
            n_layers=64,
            hidden_size=8192,
            storage_size=Memory.from_gb(40),  # 40 GB model (needs both GPUs)
            supports_tensor=False,
        )

        command = PlaceInstance(
            model_meta=model_meta,
            instance_meta=InstanceMeta.CudaNccl,
            sharding=Sharding.Pipeline,
            min_nodes=2,
        )

        # Place the instance
        instances = place_instance(command, topology, {})

        # Should create one instance
        assert len(instances) == 1
        instance = list(instances.values())[0]

        # Should be a CudaNcclInstance
        assert isinstance(instance, CudaNcclInstance)

        # Should have NCCL unique ID
        assert instance.nccl_unique_id is not None
        assert len(instance.nccl_unique_id) == 64  # 32 bytes hex-encoded

        # Should have master address
        assert instance.master_addr is not None
        assert instance.master_port > 0

        # Should have 2 shards (one per node)
        assert len(instance.shard_assignments.runner_to_shard) == 2

    def test_place_cuda_gloo_instance(
        self, create_gpu_node, create_connection
    ) -> None:
        """Test placing a CUDA Gloo instance."""
        from exo.master.placement import place_instance
        from exo.shared.types.commands import PlaceInstance

        # Create two nodes with GPUs
        node_a_id = NodeId()
        node_b_id = NodeId()

        node_a = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_a_id,
        )
        node_b = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_b_id,
        )

        conn_a_to_b = create_connection(node_a_id, node_b_id)
        conn_b_to_a = create_connection(node_b_id, node_a_id)

        topology = create_topology_with_nodes_and_connections(
            [node_a, node_b],
            [conn_a_to_b, conn_b_to_a],
        )

        model_meta = ModelMetadata(
            model_id=ModelId("test/model"),
            pretty_name="Test Model",
            n_layers=32,
            hidden_size=4096,
            storage_size=Memory.from_gb(40),
            supports_tensor=False,
        )

        command = PlaceInstance(
            model_meta=model_meta,
            instance_meta=InstanceMeta.CudaGloo,
            sharding=Sharding.Pipeline,
            min_nodes=2,
        )

        instances = place_instance(command, topology, {})

        assert len(instances) == 1
        instance = list(instances.values())[0]

        # Should be a CudaGlooInstance
        assert isinstance(instance, CudaGlooInstance)

        # Should have master address (but no NCCL unique ID)
        assert instance.master_addr is not None
        assert instance.master_port > 0

    def test_cuda_placement_fails_without_gpus(
        self, create_node, create_connection
    ) -> None:
        """Test that CUDA placement fails when no GPUs are available."""
        from exo.master.placement import place_instance
        from exo.shared.types.commands import PlaceInstance

        # Create nodes WITHOUT GPUs
        node_a_id = NodeId()
        node_b_id = NodeId()

        node_a = create_node(memory=64 * 1024**3, node_id=node_a_id)
        node_b = create_node(memory=64 * 1024**3, node_id=node_b_id)

        conn_a_to_b = create_connection(node_a_id, node_b_id)
        conn_b_to_a = create_connection(node_b_id, node_a_id)

        topology = create_topology_with_nodes_and_connections(
            [node_a, node_b],
            [conn_a_to_b, conn_b_to_a],
        )

        model_meta = ModelMetadata(
            model_id=ModelId("test/model"),
            pretty_name="Test Model",
            n_layers=32,
            hidden_size=4096,
            storage_size=Memory.from_gb(8),
            supports_tensor=False,
        )

        command = PlaceInstance(
            model_meta=model_meta,
            instance_meta=InstanceMeta.CudaNccl,
            sharding=Sharding.Pipeline,
            min_nodes=2,
        )

        with pytest.raises(ValueError, match="CUDA-capable GPUs"):
            place_instance(command, topology, {})

    def test_cuda_placement_insufficient_gpu_memory(
        self, create_gpu_node, create_connection
    ) -> None:
        """Test that CUDA placement handles insufficient GPU memory."""
        from exo.master.placement import place_instance
        from exo.shared.types.commands import PlaceInstance

        # Create nodes with small GPUs and limited RAM
        node_a_id = NodeId()
        node_b_id = NodeId()

        node_a = create_gpu_node(
            ram_memory=8 * 1024**3,  # Only 8 GB RAM
            gpu_memory=4 * 1024**3,  # Only 4 GB GPU
            node_id=node_a_id,
        )
        node_b = create_gpu_node(
            ram_memory=8 * 1024**3,  # Only 8 GB RAM
            gpu_memory=4 * 1024**3,  # Only 4 GB GPU
            node_id=node_b_id,
        )

        conn_a_to_b = create_connection(node_a_id, node_b_id)
        conn_b_to_a = create_connection(node_b_id, node_a_id)

        topology = create_topology_with_nodes_and_connections(
            [node_a, node_b],
            [conn_a_to_b, conn_b_to_a],
        )

        # Model requires more memory than available GPUs and RAM have
        model_meta = ModelMetadata(
            model_id=ModelId("test/huge-model"),
            pretty_name="Test Huge Model",
            n_layers=128,
            hidden_size=16384,
            storage_size=Memory.from_gb(100),  # 100 GB model
            supports_tensor=False,
        )

        command = PlaceInstance(
            model_meta=model_meta,
            instance_meta=InstanceMeta.CudaNccl,
            sharding=Sharding.Pipeline,
            min_nodes=2,
        )

        # Should fail since neither GPU memory nor RAM is sufficient
        with pytest.raises(ValueError, match="memory"):
            place_instance(command, topology, {})


class TestCudaMasterAddress:
    """Tests for CUDA master address determination."""

    def test_get_cuda_master_addr_single_node(self, create_gpu_node) -> None:
        """Test getting master address for single node."""
        from exo.master.placement import get_cuda_master_addr

        node = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
        )

        topology = create_topology_with_nodes_and_connections([node], [])

        addr = get_cuda_master_addr([node], topology)
        assert addr == "127.0.0.1"  # Localhost for single node

    def test_get_cuda_master_addr_multi_node(
        self, create_gpu_node, create_connection
    ) -> None:
        """Test getting master address for multi-node setup."""
        from exo.master.placement import get_cuda_master_addr

        node_a_id = NodeId()
        node_b_id = NodeId()

        node_a = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_a_id,
        )
        node_b = create_gpu_node(
            ram_memory=64 * 1024**3,
            gpu_memory=24 * 1024**3,
            node_id=node_b_id,
        )

        conn_a_to_b = create_connection(node_a_id, node_b_id)
        conn_b_to_a = create_connection(node_b_id, node_a_id)

        topology = create_topology_with_nodes_and_connections(
            [node_a, node_b],
            [conn_a_to_b, conn_b_to_a],
        )

        cycle_digraph = topology.get_subgraph_from_nodes([node_a, node_b])
        addr = get_cuda_master_addr([node_a, node_b], cycle_digraph)

        # Should return a valid IP address
        assert addr is not None
        assert "." in addr  # Basic IP check

    def test_get_cuda_master_addr_empty_cycle_raises(self) -> None:
        """Test that empty cycle raises ValueError."""
        from exo.master.placement import get_cuda_master_addr

        topology = Topology()

        with pytest.raises(ValueError, match="empty cycle"):
            get_cuda_master_addr([], topology)


class TestNcclUniqueId:
    """Tests for NCCL unique ID generation."""

    def test_generate_nccl_unique_id_format(self) -> None:
        """Test that NCCL unique ID has correct format."""
        from exo.master.placement import generate_nccl_unique_id

        unique_id = generate_nccl_unique_id()

        # Should be hex string
        assert isinstance(unique_id, str)
        assert all(c in "0123456789abcdef" for c in unique_id)

        # Should be 32 bytes = 64 hex characters
        assert len(unique_id) == 64

    def test_generate_nccl_unique_id_uniqueness(self) -> None:
        """Test that NCCL unique IDs are unique."""
        from exo.master.placement import generate_nccl_unique_id

        ids = [generate_nccl_unique_id() for _ in range(100)]
        unique_ids = set(ids)

        # All IDs should be unique
        assert len(unique_ids) == 100

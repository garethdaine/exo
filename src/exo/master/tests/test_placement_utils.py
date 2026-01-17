from typing import Callable

import pytest

from exo.master.placement_utils import (
    filter_cycles_by_compute_capability,
    filter_cycles_by_cuda_capability,
    filter_cycles_by_gpu_memory,
    filter_cycles_by_memory,
    get_cuda_device_ids_for_cycle,
    get_hosts_from_subgraph,
    get_mlx_jaccl_coordinators,
    get_shard_assignments,
    get_smallest_cycles,
    rank_cycles_by_gpu_quality,
)
from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId, ModelMetadata
from exo.shared.types.profiling import NetworkInterfaceInfo, NodePerformanceProfile
from exo.shared.types.topology import Connection, NodeInfo
from exo.shared.types.worker.shards import Sharding


@pytest.fixture
def topology() -> Topology:
    topology = Topology()
    return topology


def test_filter_cycles_by_memory(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()

    node1 = create_node(1000 * 1024, node1_id)
    node2 = create_node(1000 * 1024, node2_id)

    topology.add_node(node1)
    topology.add_node(node2)

    connection1 = create_connection(node1_id, node2_id)
    connection2 = create_connection(node2_id, node1_id)

    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = topology.get_cycles()
    assert len(cycles) == 1
    assert len(cycles[0]) == 2

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, Memory.from_bytes(1))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 2
    assert set(n.node_id for n in filtered_cycles[0]) == {node1_id, node2_id}


def test_filter_cycles_by_insufficient_memory(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # arrange
    node1_id = NodeId()
    node2_id = NodeId()

    node1 = create_node(1000 * 1024, node1_id)
    node2 = create_node(1000 * 1024, node2_id)

    topology.add_node(node1)
    topology.add_node(node2)

    connection1 = create_connection(node1_id, node2_id)
    connection2 = create_connection(node2_id, node1_id)

    topology.add_connection(connection1)
    topology.add_connection(connection2)

    # act
    filtered_cycles = filter_cycles_by_memory(
        topology.get_cycles(), Memory.from_kb(2001)
    )

    # assert
    assert len(filtered_cycles) == 0


def test_filter_multiple_cycles_by_memory(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    node_a = create_node(500 * 1024, node_a_id)
    node_b = create_node(500 * 1024, node_b_id)
    node_c = create_node(1000 * 1024, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))

    topology.add_connection(create_connection(node_a_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_b_id))

    cycles = topology.get_cycles()

    # act
    filtered_cycles = filter_cycles_by_memory(cycles, Memory.from_kb(1500))

    # assert
    assert len(filtered_cycles) == 1
    assert len(filtered_cycles[0]) == 3
    assert set(n.node_id for n in filtered_cycles[0]) == {
        node_a_id,
        node_b_id,
        node_c_id,
    }


def test_get_smallest_cycles(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    node_a = create_node(500 * 1024, node_a_id)
    node_b = create_node(500 * 1024, node_b_id)
    node_c = create_node(1000 * 1024, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_a_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))

    # act
    smallest_cycles = get_smallest_cycles(topology.get_cycles())

    # assert
    assert len(smallest_cycles) == 1
    assert len(smallest_cycles[0]) == 2
    assert set(n.node_id for n in smallest_cycles[0]) == {node_a_id, node_b_id}


@pytest.mark.parametrize(
    "available_memory,total_layers,expected_layers",
    [
        ((500, 500, 1000), 12, (3, 3, 6)),
        ((500, 500, 500), 12, (4, 4, 4)),
        ((312, 518, 1024), 12, (2, 3, 7)),
    ],
)
def test_get_shard_assignments(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
    available_memory: tuple[int, int, int],
    total_layers: int,
    expected_layers: tuple[int, int, int],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    node_a = create_node(available_memory[0] * 1024, node_a_id)
    node_b = create_node(available_memory[1] * 1024, node_b_id)
    node_c = create_node(available_memory[2] * 1024, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id))
    topology.add_connection(create_connection(node_b_id, node_c_id))
    topology.add_connection(create_connection(node_c_id, node_a_id))
    topology.add_connection(create_connection(node_b_id, node_a_id))

    model_meta = ModelMetadata(
        model_id=ModelId("test-model"),
        pretty_name="Test Model",
        n_layers=total_layers,
        storage_size=Memory.from_kb(1000),
        hidden_size=1000,
        supports_tensor=True,
    )
    cycles = topology.get_cycles()
    selected_cycle = cycles[0]

    # act
    shard_assignments = get_shard_assignments(
        model_meta, selected_cycle, Sharding.Pipeline
    )

    # assert
    runner_id_a = shard_assignments.node_to_runner[node_a_id]
    runner_id_b = shard_assignments.node_to_runner[node_b_id]
    runner_id_c = shard_assignments.node_to_runner[node_c_id]
    assert (
        shard_assignments.runner_to_shard[runner_id_c].end_layer
        - shard_assignments.runner_to_shard[runner_id_c].start_layer
        == expected_layers[2]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_a].end_layer
        - shard_assignments.runner_to_shard[runner_id_a].start_layer
        == expected_layers[0]
    )
    assert (
        shard_assignments.runner_to_shard[runner_id_b].end_layer
        - shard_assignments.runner_to_shard[runner_id_b].start_layer
        == expected_layers[1]
    )


def test_get_hosts_from_subgraph(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId, int | None], Connection],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    node_a = create_node(500, node_a_id)
    node_b = create_node(500, node_b_id)
    node_c = create_node(1000, node_c_id)

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(create_connection(node_a_id, node_b_id, 5001))
    topology.add_connection(create_connection(node_b_id, node_c_id, 5002))
    topology.add_connection(create_connection(node_c_id, node_a_id, 5003))
    topology.add_connection(create_connection(node_b_id, node_a_id, 5004))

    # act
    hosts = get_hosts_from_subgraph(topology)

    # assert
    assert len(hosts) == 3
    expected_hosts = [
        Host(ip=("169.254.0.2"), port=5001),
        Host(ip=("169.254.0.3"), port=5002),
        Host(ip=("169.254.0.4"), port=5003),
    ]
    for expected_host in expected_hosts:
        assert expected_host in hosts


def test_get_mlx_jaccl_coordinators(
    topology: Topology,
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId, int | None], Connection],
):
    # arrange
    node_a_id = NodeId()
    node_b_id = NodeId()
    node_c_id = NodeId()

    node_a = create_node(500 * 1024, node_a_id)
    node_b = create_node(500 * 1024, node_b_id)
    node_c = create_node(1000 * 1024, node_c_id)

    conn_a_b = create_connection(node_a_id, node_b_id, 5001)
    conn_b_a = create_connection(node_b_id, node_a_id, 5002)
    conn_b_c = create_connection(node_b_id, node_c_id, 5003)
    conn_c_b = create_connection(node_c_id, node_b_id, 5004)
    conn_c_a = create_connection(node_c_id, node_a_id, 5005)
    conn_a_c = create_connection(node_a_id, node_c_id, 5006)

    # Update node profiles with network interfaces before adding to topology
    assert node_a.node_profile is not None
    assert node_b.node_profile is not None
    assert node_c.node_profile is not None

    node_a.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_a.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_a_b.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_a_c.send_back_multiaddr.ip_address,
            ),
        ],
        system=node_a.node_profile.system,
    )
    node_b.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_b.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_b_a.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_b_c.send_back_multiaddr.ip_address,
            ),
        ],
        system=node_b.node_profile.system,
    )
    node_c.node_profile = NodePerformanceProfile(
        model_id="test",
        chip_id="test",
        friendly_name="test",
        memory=node_c.node_profile.memory,
        network_interfaces=[
            NetworkInterfaceInfo(
                name="en3",
                ip_address=conn_c_b.send_back_multiaddr.ip_address,
            ),
            NetworkInterfaceInfo(
                name="en4",
                ip_address=conn_c_a.send_back_multiaddr.ip_address,
            ),
        ],
        system=node_c.node_profile.system,
    )

    topology.add_node(node_a)
    topology.add_node(node_b)
    topology.add_node(node_c)

    topology.add_connection(conn_a_b)
    topology.add_connection(conn_b_a)
    topology.add_connection(conn_b_c)
    topology.add_connection(conn_c_b)
    topology.add_connection(conn_c_a)
    topology.add_connection(conn_a_c)

    cycle = [node_a, node_b, node_c]

    # act
    coordinators = get_mlx_jaccl_coordinators(
        cycle, coordinator_port=5000, cycle_digraph=topology
    )

    # assert
    assert len(coordinators) == 3
    assert node_a_id in coordinators
    assert node_b_id in coordinators
    assert node_c_id in coordinators

    # All coordinators should have IP:PORT format
    for node_id, coordinator in coordinators.items():
        assert ":" in coordinator, (
            f"Coordinator for {node_id} should have ':' separator"
        )

    # Verify port is correct
    for node_id, coordinator in coordinators.items():
        assert coordinator.endswith(":5000"), (
            f"Coordinator for {node_id} should use port 5000"
        )

    # Rank 0 (node_a) treats this as the listen socket so should listen on all
    # IPs
    assert coordinators[node_a_id].startswith("0.0.0.0:"), (
        "Rank 0 node should use localhost as coordinator"
    )

    # Non-rank-0 nodes should use the specific IP from their connection to rank 0
    # node_b uses the IP from conn_b_a (node_b -> node_a)
    assert coordinators[node_b_id] == (
        f"{conn_b_a.send_back_multiaddr.ip_address}:5000"
    ), "node_b should use the IP from conn_b_a"

    # node_c uses the IP from conn_c_a (node_c -> node_a)
    assert coordinators[node_c_id] == (
        f"{conn_c_a.send_back_multiaddr.ip_address}:5000"
    ), "node_c should use the IP from conn_c_a"


# =============================================================================
# GPU-Aware Placement Tests (Phase 7)
# =============================================================================


def test_filter_cycles_by_gpu_memory(
    topology: Topology,
    create_gpu_node: Callable[..., NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    """Test filtering cycles by GPU memory."""
    # arrange - create nodes with different GPU memory
    node1_id = NodeId()
    node2_id = NodeId()

    # 16GB GPU memory each = 32GB total
    node1 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node1_id,
    )
    node2 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node2_id,
    )

    topology.add_node(node1)
    topology.add_node(node2)

    connection1 = create_connection(node1_id, node2_id)
    connection2 = create_connection(node2_id, node1_id)

    topology.add_connection(connection1)
    topology.add_connection(connection2)

    cycles = topology.get_cycles()
    assert len(cycles) == 1

    # act - filter with memory requirement less than total GPU memory
    filtered = filter_cycles_by_gpu_memory(cycles, Memory.from_gb(20))

    # assert - cycle should pass
    assert len(filtered) == 1

    # act - filter with memory requirement more than total GPU memory
    filtered = filter_cycles_by_gpu_memory(cycles, Memory.from_gb(40))

    # assert - cycle should not pass
    assert len(filtered) == 0


def test_filter_cycles_by_cuda_capability(
    topology: Topology,
    create_gpu_node: Callable[..., NodeInfo],
    create_node: Callable[[int, NodeId | None], NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    """Test filtering cycles to only include those with CUDA GPUs."""
    # arrange - one node with GPU, one without
    node1_id = NodeId()
    node2_id = NodeId()
    node3_id = NodeId()

    node1 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node1_id,
    )
    node2 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node2_id,
    )
    # node3 has no GPU (uses create_node which doesn't add GPU profiles)
    node3 = create_node(64 * 1024**3, node3_id)

    topology.add_node(node1)
    topology.add_node(node2)
    topology.add_node(node3)

    # Create two cycles: (node1, node2) and (node1, node2, node3)
    topology.add_connection(create_connection(node1_id, node2_id))
    topology.add_connection(create_connection(node2_id, node1_id))
    topology.add_connection(create_connection(node2_id, node3_id))
    topology.add_connection(create_connection(node3_id, node1_id))

    cycles = topology.get_cycles()

    # act
    filtered = filter_cycles_by_cuda_capability(cycles)

    # assert - only cycles with all CUDA nodes should pass
    assert len(filtered) == 1
    assert len(filtered[0]) == 2  # The 2-node cycle with both GPU nodes


def test_filter_cycles_by_compute_capability(
    topology: Topology,
    create_gpu_node: Callable[..., NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    """Test filtering cycles by minimum compute capability."""
    # arrange - nodes with different compute capabilities
    node1_id = NodeId()
    node2_id = NodeId()

    node1 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node1_id,
        compute_capability="8.0",  # Ampere
    )
    node2 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node2_id,
        compute_capability="7.5",  # Turing
    )

    topology.add_node(node1)
    topology.add_node(node2)

    topology.add_connection(create_connection(node1_id, node2_id))
    topology.add_connection(create_connection(node2_id, node1_id))

    cycles = topology.get_cycles()
    assert len(cycles) == 1

    # act - require SM 7.0 (both nodes should pass)
    filtered = filter_cycles_by_compute_capability(cycles, "7.0")
    assert len(filtered) == 1

    # act - require SM 8.0 (only node1 passes, but cycle needs all nodes)
    filtered = filter_cycles_by_compute_capability(cycles, "8.0")
    assert len(filtered) == 0


def test_get_cuda_device_ids_for_cycle(
    create_gpu_node: Callable[..., NodeInfo],
):
    """Test getting device IDs for nodes in a cycle."""
    node1_id = NodeId()
    node2_id = NodeId()

    # Node with 2 GPUs
    node1 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node1_id,
        num_gpus=2,
    )
    # Node with 4 GPUs
    node2 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=16 * 1024**3,
        node_id=node2_id,
        num_gpus=4,
    )

    cycle = [node1, node2]

    # act
    device_ids = get_cuda_device_ids_for_cycle(cycle)

    # assert
    assert device_ids[node1_id] == [0, 1]
    assert device_ids[node2_id] == [0, 1, 2, 3]


def test_rank_cycles_by_gpu_quality(
    topology: Topology,
    create_gpu_node: Callable[..., NodeInfo],
    create_connection: Callable[[NodeId, NodeId], Connection],
):
    """Test ranking cycles by GPU quality (memory, NVLink)."""
    # arrange - create two separate cycles with different GPU quality
    node1_id = NodeId()
    node2_id = NodeId()
    node3_id = NodeId()
    node4_id = NodeId()

    # First cycle: lower memory, no NVLink
    node1 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=8 * 1024**3,  # 8GB
        node_id=node1_id,
        nvlink_supported=False,
    )
    node2 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=8 * 1024**3,  # 8GB
        node_id=node2_id,
        nvlink_supported=False,
    )

    # Second cycle: higher memory, with NVLink
    node3 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=24 * 1024**3,  # 24GB
        node_id=node3_id,
        nvlink_supported=True,
    )
    node4 = create_gpu_node(
        ram_memory=64 * 1024**3,
        gpu_memory=24 * 1024**3,  # 24GB
        node_id=node4_id,
        nvlink_supported=True,
    )

    # Create two separate 2-node cycles
    cycles = [[node1, node2], [node3, node4]]

    # act
    ranked = rank_cycles_by_gpu_quality(cycles)

    # assert - higher memory cycle should be first
    assert len(ranked) == 2
    # The cycle with 48GB total should come first
    assert node3 in ranked[0] or node4 in ranked[0]

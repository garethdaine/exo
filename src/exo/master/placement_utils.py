from collections.abc import Generator
from typing import TypeGuard, cast

from loguru import logger
from pydantic import BaseModel

from exo.shared.topology import Topology
from exo.shared.types.common import Host, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelMetadata
from exo.shared.types.profiling import NodePerformanceProfile
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.runners import RunnerId, ShardAssignments
from exo.shared.types.worker.shards import (
    PipelineShardMetadata,
    Sharding,
    ShardMetadata,
    TensorShardMetadata,
)


class NodeWithProfile(BaseModel):
    node_id: NodeId
    node_profile: NodePerformanceProfile


def narrow_all_nodes(nodes: list[NodeInfo]) -> TypeGuard[list[NodeWithProfile]]:
    return all(node.node_profile is not None for node in nodes)


def filter_cycles_by_memory(
    cycles: list[list[NodeInfo]], required_memory: Memory
) -> list[list[NodeInfo]]:
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_mem = sum(
            (node.node_profile.memory.ram_available for node in cycle), start=Memory()
        )
        if total_mem >= required_memory:
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def filter_cycles_by_gpu_memory(
    cycles: list[list[NodeInfo]], required_memory: Memory
) -> list[list[NodeInfo]]:
    """Filter cycles to those with sufficient total GPU memory.

    This is used for CUDA instances where model weights must fit in GPU VRAM.
    A cycle passes if the sum of available GPU memory across all nodes
    is at least the required memory.

    Args:
        cycles: List of node cycles to filter.
        required_memory: Minimum total GPU memory needed.

    Returns:
        Cycles that have sufficient combined GPU memory.
    """
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        total_gpu_mem = sum(
            node.node_profile.available_gpu_memory_bytes for node in cycle
        )
        if total_gpu_mem >= required_memory.in_bytes:
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def filter_cycles_by_cuda_capability(
    cycles: list[list[NodeInfo]],
) -> list[list[NodeInfo]]:
    """Filter cycles to those where all nodes have CUDA GPUs.

    This ensures that a cycle selected for CUDA instances actually has
    NVIDIA GPUs available on all participating nodes.

    Args:
        cycles: List of node cycles to filter.

    Returns:
        Cycles where all nodes have at least one CUDA GPU.
    """
    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        # All nodes must have at least one CUDA GPU
        if all(node.node_profile.has_cuda_gpus for node in cycle):
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def filter_cycles_by_compute_capability(
    cycles: list[list[NodeInfo]], min_compute_capability: str
) -> list[list[NodeInfo]]:
    """Filter cycles to those where all GPUs meet minimum compute capability.

    Some models require specific CUDA compute capabilities (e.g., SM 8.0 for
    efficient BFloat16 support). This filter ensures all GPUs in a cycle
    meet the minimum requirement.

    Args:
        cycles: List of node cycles to filter.
        min_compute_capability: Minimum compute capability (e.g., "8.0").

    Returns:
        Cycles where all GPUs have at least the minimum compute capability.
    """
    min_major, min_minor = _parse_compute_capability(min_compute_capability)
    if min_major is None:
        return cycles  # No filtering if we can't parse the requirement

    filtered_cycles: list[list[NodeInfo]] = []
    for cycle in cycles:
        if not narrow_all_nodes(cycle):
            continue

        cycle_meets_requirement = True
        for node in cycle:
            for gpu in node.node_profile.gpu_profiles:
                if gpu.compute_capability is None:
                    cycle_meets_requirement = False
                    break
                gpu_major, gpu_minor = _parse_compute_capability(gpu.compute_capability)
                if gpu_major is None:
                    cycle_meets_requirement = False
                    break
                if (gpu_major, gpu_minor) < (min_major, min_minor):
                    cycle_meets_requirement = False
                    break
            if not cycle_meets_requirement:
                break

        if cycle_meets_requirement:
            filtered_cycles.append(cast(list[NodeInfo], cycle))
    return filtered_cycles


def _parse_compute_capability(cc: str) -> tuple[int | None, int | None]:
    """Parse a compute capability string like '8.0' into (major, minor)."""
    try:
        parts = cc.split(".")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except (ValueError, AttributeError):
        pass
    return None, None


def get_cuda_device_ids_for_cycle(cycle: list[NodeInfo]) -> dict[NodeId, list[int]]:
    """Get the GPU device indices to use on each node in a cycle.

    Returns a mapping from node ID to list of device indices (e.g., [0, 1]
    for a node with 2 GPUs).

    Args:
        cycle: List of nodes in the selected cycle.

    Returns:
        Mapping from node ID to list of GPU device indices.
    """
    device_ids: dict[NodeId, list[int]] = {}
    for node in cycle:
        if node.node_profile is None:
            device_ids[node.node_id] = []
            continue

        # Get indices of all CUDA GPUs on this node
        cuda_indices = [
            gpu.device_index
            for gpu in node.node_profile.gpu_profiles
            if gpu.accelerator_type.value == "nvidia_cuda"
        ]
        device_ids[node.node_id] = cuda_indices
    return device_ids


def rank_cycles_by_gpu_quality(
    cycles: list[list[NodeInfo]],
) -> list[list[NodeInfo]]:
    """Rank cycles by GPU quality for optimal placement.

    Cycles are sorted by:
    1. Total available GPU memory (higher is better)
    2. NVLink support (cycles with NVLink preferred)
    3. Number of GPUs (more GPUs preferred for parallelism)

    Args:
        cycles: List of node cycles to rank.

    Returns:
        Cycles sorted by GPU quality (best first).
    """

    def score_cycle(cycle: list[NodeInfo]) -> tuple[int, int, int]:
        total_gpu_mem = 0
        nvlink_count = 0
        gpu_count = 0

        for node in cycle:
            if node.node_profile is None:
                continue
            for gpu in node.node_profile.gpu_profiles:
                total_gpu_mem += gpu.memory.free_bytes
                if gpu.nvlink_supported:
                    nvlink_count += 1
                gpu_count += 1

        return (total_gpu_mem, nvlink_count, gpu_count)

    return sorted(cycles, key=score_cycle, reverse=True)


def get_smallest_cycles(cycles: list[list[NodeInfo]]) -> list[list[NodeInfo]]:
    min_nodes = min(len(cycle) for cycle in cycles)
    return [cycle for cycle in cycles if len(cycle) == min_nodes]


def get_shard_assignments_for_pipeline_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
):
    cycle_memory = sum(
        (node.node_profile.memory.ram_available for node in selected_cycle),
        start=Memory(),
    )
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    layers_assigned = 0
    for i, node in enumerate(selected_cycle):
        if i == len(selected_cycle) - 1:
            node_layers = total_layers - layers_assigned
        else:
            node_layers = round(
                total_layers
                * (
                    node.node_profile.memory.ram_available.in_bytes
                    / cycle_memory.in_bytes
                )
            )
            node_layers = max(1, node_layers)

        runner_id = RunnerId()

        shard = PipelineShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=layers_assigned,
            end_layer=layers_assigned + node_layers,
            n_layers=total_layers,
        )

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id
        layers_assigned += node_layers

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments_for_tensor_parallel(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeWithProfile],
):
    total_layers = model_meta.n_layers
    world_size = len(selected_cycle)
    runner_to_shard: dict[RunnerId, ShardMetadata] = {}
    node_to_runner: dict[NodeId, RunnerId] = {}

    for i, node in enumerate(selected_cycle):
        shard = TensorShardMetadata(
            model_meta=model_meta,
            device_rank=i,
            world_size=world_size,
            start_layer=0,
            end_layer=total_layers,
            n_layers=total_layers,
        )

        runner_id = RunnerId()

        runner_to_shard[runner_id] = shard
        node_to_runner[node.node_id] = runner_id

    shard_assignments = ShardAssignments(
        model_id=model_meta.model_id,
        runner_to_shard=runner_to_shard,
        node_to_runner=node_to_runner,
    )

    return shard_assignments


def get_shard_assignments(
    model_meta: ModelMetadata,
    selected_cycle: list[NodeInfo],
    sharding: Sharding,
) -> ShardAssignments:
    if not narrow_all_nodes(selected_cycle):
        raise ValueError("All nodes must have profiles to create shard assignments")
    match sharding:
        case Sharding.Pipeline:
            return get_shard_assignments_for_pipeline_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )
        case Sharding.Tensor:
            return get_shard_assignments_for_tensor_parallel(
                model_meta=model_meta,
                selected_cycle=selected_cycle,
            )


def get_hosts_from_subgraph(cycle_digraph: Topology) -> list[Host]:
    cycles = cycle_digraph.get_cycles()
    expected_length = len(list(cycle_digraph.list_nodes()))
    cycles = [cycle for cycle in cycles if len(cycle) == expected_length]
    if not cycles:
        if expected_length > 1:
            logger.warning(
                f"No cycles of length {expected_length} found even though chosen subgraph contained {expected_length} nodes"
            )
        return []

    get_thunderbolt = False
    if cycle_digraph.is_thunderbolt_cycle(cycles[0]):
        get_thunderbolt = True

    logger.info(f"Using thunderbolt cycle: {get_thunderbolt}")

    cycle = cycles[0]
    hosts: list[Host] = []
    for i in range(len(cycle)):
        current_node = cycle[i]
        next_node = cycle[(i + 1) % len(cycle)]

        for connection in cycle_digraph.list_connections():
            if (
                connection.local_node_id == current_node.node_id
                and connection.send_back_node_id == next_node.node_id
            ):
                if get_thunderbolt and not connection.is_thunderbolt():
                    continue
                assert connection.send_back_multiaddr is not None
                host = Host(
                    ip=connection.send_back_multiaddr.ip_address,
                    port=connection.send_back_multiaddr.port,
                )
                hosts.append(host)
                break

    return hosts


def get_mlx_ibv_devices_matrix(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
) -> list[list[str | None]]:
    """Build connectivity matrix mapping device i to device j via RDMA interface names.

    The matrix element [i][j] contains the interface name on device i that connects
    to device j, or None if no connection exists or no interface name is found.
    Diagonal elements are always None.
    """
    num_nodes = len(selected_cycle)
    matrix: list[list[str | None]] = [
        [None for _ in range(num_nodes)] for _ in range(num_nodes)
    ]

    for i, node_i in enumerate(selected_cycle):
        for j, node_j in enumerate(selected_cycle):
            if i == j:
                continue

            # Find the IP J uses to talk to I
            for connection_ip, _ in _find_connection_ip(node_j, node_i, cycle_digraph):
                # This is a local IP on I, which is attached to an interface: find that interface
                if interface_name := _find_rdma_interface_name_for_ip(
                    connection_ip, node_i
                ):
                    matrix[i][j] = interface_name
                    logger.info(
                        f"Interface name for {connection_ip} on {node_i.node_id}: {interface_name}"
                    )
                    break
            else:
                logger.warning(
                    f"Failed to find interface name between {node_i.node_id} and {node_j.node_id}"
                )
                raise ValueError(
                    "Current ibv backend requires all-to-all rdma connections"
                )

    return matrix


def _find_connection_ip(
    node_i: NodeInfo,
    node_j: NodeInfo,
    cycle_digraph: Topology,
) -> Generator[tuple[str, bool]]:
    """Find all IP addresses that connect node i to node j, with thunderbolt flag."""
    for connection in cycle_digraph.list_connections():
        if (
            connection.local_node_id == node_i.node_id
            and connection.send_back_node_id == node_j.node_id
        ):
            yield connection.send_back_multiaddr.ip_address, connection.is_thunderbolt()


def _find_rdma_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    if node_info.node_profile is None:
        return None

    logger.info(f"Searching {node_info.node_id} for ip {ip_address}:")
    for interface in node_info.node_profile.network_interfaces:
        if interface.name not in ["en2", "en3", "en4", "en5", "en6", "en7"]:
            continue
        logger.info(f" | {interface.name}: {interface.ip_address}")
        if interface.ip_address != ip_address:
            continue

        logger.info("Found")
        return f"rdma_{interface.name}"

    return None


def _find_interface_name_for_ip(
    ip_address: str,
    node_info: NodeInfo,
) -> str | None:
    """Find the interface name for an IP address on a node (any interface)."""
    if node_info.node_profile is None:
        return None

    for interface in node_info.node_profile.network_interfaces:
        if interface.ip_address == ip_address:
            return interface.name

    return None


def _find_ip_prioritised(
    node: NodeInfo, other_node: NodeInfo, cycle_digraph: Topology
) -> str | None:
    # TODO: Actually prioritize in the correct Ethernet > Wifi > Non-TB > TB order.
    """Find an IP address between nodes with prioritization.

    Priority order:
    1. en0 (Ethernet on Mac Studio, WiFi on MacBook)
    2. en1 (WiFi on Mac Studio, Ethernet on MacBook)
    3. Non-Thunderbolt connections
    4. Any other IP address
    """
    ips = list(_find_connection_ip(node, other_node, cycle_digraph))
    # We expect a unique iface -> ip mapping
    iface_map = {_find_interface_name_for_ip(ip, other_node): ip for ip, _ in ips}

    en0_ip = iface_map.get("en0")
    if en0_ip:
        return en0_ip

    en1_ip = iface_map.get("en1")
    if en1_ip:
        return en1_ip

    non_thunderbolt_ip = next(
        (ip for (ip, is_thunderbolt) in ips if not is_thunderbolt), None
    )

    if non_thunderbolt_ip:
        return non_thunderbolt_ip

    if ips:
        return ips[0][0]

    return None


def get_mlx_ring_hosts_by_node(
    selected_cycle: list[NodeInfo],
    cycle_digraph: Topology,
    ephemeral_port: int,
) -> dict[NodeId, list[Host]]:
    """Generate per-node host lists for MLX ring backend.

    Each node gets a list where:
    - Self position: Host(ip="0.0.0.0", port=ephemeral_port)
    - Left/right neighbors: actual connection IPs
    - Non-neighbors: Host(ip="198.51.100.1", port=0) placeholder (RFC 5737 TEST-NET-2)
    """
    world_size = len(selected_cycle)
    if world_size == 0:
        return {}

    hosts_by_node: dict[NodeId, list[Host]] = {}

    for rank, node in enumerate(selected_cycle):
        node_id = node.node_id
        left_rank = (rank - 1) % world_size
        right_rank = (rank + 1) % world_size

        hosts_for_node: list[Host] = []

        for idx, other_node in enumerate(selected_cycle):
            if idx == rank:
                hosts_for_node.append(Host(ip="0.0.0.0", port=ephemeral_port))
                continue

            if idx not in {left_rank, right_rank}:
                # Placeholder IP from RFC 5737 TEST-NET-2
                hosts_for_node.append(Host(ip="198.51.100.1", port=0))
                continue

            connection_ip = _find_ip_prioritised(node, other_node, cycle_digraph)
            if connection_ip is None:
                logger.warning(
                    f"Failed to find prioritised connection IP between {node_id} and {other_node.node_id}"
                )
                raise ValueError(
                    "MLX ring backend requires connectivity between neighbouring nodes"
                )

            hosts_for_node.append(Host(ip=connection_ip, port=ephemeral_port))

        hosts_by_node[node_id] = hosts_for_node

    return hosts_by_node


def get_mlx_jaccl_coordinators(
    selected_cycle: list[NodeInfo],
    coordinator_port: int,
    cycle_digraph: Topology,
) -> dict[NodeId, str]:
    """Get the coordinator addresses for MLX Jaccl (rank 0 device).

    Select an IP address that each node can reach for the rank 0 node. Returns
    address in format "X.X.X.X:PORT" per node.
    """
    rank_0_node = selected_cycle[0]
    logger.debug(f"Selecting coordinator from rank 0 node: {rank_0_node.node_id}")

    def get_ip_for_node(n: NodeInfo) -> str:
        if n.node_id == rank_0_node.node_id:
            return "0.0.0.0"

        ip = _find_ip_prioritised(n, rank_0_node, cycle_digraph)
        if ip:
            return ip

        logger.warning(
            f"Failed to find directly connected ip between {n.node_id} and {rank_0_node.node_id}"
        )
        raise ValueError("Current ibv backend requires all-to-all rdma connections")

    return {
        n.node_id: f"{get_ip_for_node(n)}:{coordinator_port}" for n in selected_cycle
    }

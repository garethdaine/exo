import random
import secrets
from collections.abc import Mapping
from copy import deepcopy
from typing import Sequence

from loguru import logger

from exo.master.placement_utils import (
    filter_cycles_by_cuda_capability,
    filter_cycles_by_gpu_memory,
    filter_cycles_by_memory,
    get_cuda_device_ids_for_cycle,
    get_mlx_ibv_devices_matrix,
    get_mlx_jaccl_coordinators,
    get_mlx_ring_hosts_by_node,
    get_shard_assignments,
    get_smallest_cycles,
    rank_cycles_by_gpu_quality,
)
from exo.shared.topology import Topology
from exo.shared.types.commands import (
    CreateInstance,
    DeleteInstance,
    PlaceInstance,
)
from exo.shared.types.events import Event, InstanceCreated, InstanceDeleted
from exo.shared.types.memory import Memory
from exo.shared.types.models import ModelId
from exo.shared.types.topology import NodeInfo
from exo.shared.types.worker.instances import (
    CudaGlooInstance,
    CudaNcclInstance,
    Instance,
    InstanceId,
    InstanceMeta,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.shards import Sharding


def random_ephemeral_port() -> int:
    port = random.randint(49153, 65535)
    return port - 1 if port <= 52415 else 52414


def generate_nccl_unique_id() -> str:
    """Generate a unique ID for NCCL communicator initialization.

    NCCL requires a unique identifier to establish a communicator group.
    This ID must be shared among all processes that will participate in
    the collective operations.

    The actual NCCL unique ID is typically 128 bytes (1024 bits), generated
    by the rank-0 process using ncclGetUniqueId(). Here we generate a
    placeholder that will be used to coordinate the actual ID generation
    on the rank-0 worker.

    Returns:
        A hex-encoded 32-byte random string as placeholder for NCCL ID.
    """
    return secrets.token_hex(32)


def get_cuda_master_addr(
    selected_cycle: list[NodeInfo], cycle_digraph: Topology
) -> str:
    """Get the master address for CUDA distributed initialization.

    The master address is an IP address of the rank-0 node that all other
    nodes can reach. For NCCL/Gloo initialization, workers use this address
    to coordinate communicator setup.

    Args:
        selected_cycle: List of nodes in the selected cycle (rank-0 is first).
        cycle_digraph: Subgraph topology containing only the selected nodes.

    Returns:
        IP address string for the rank-0 node.
    """
    if not selected_cycle:
        raise ValueError("Cannot get master address from empty cycle")

    rank_0_node = selected_cycle[0]

    # For single-node case, use localhost
    if len(selected_cycle) == 1:
        return "127.0.0.1"

    # Find an IP address on rank-0 that other nodes can reach
    # Look at connections from rank-1 to rank-0 to find a reachable IP
    rank_1_node = selected_cycle[1]

    for connection in cycle_digraph.list_connections():
        if (
            connection.local_node_id == rank_1_node.node_id
            and connection.send_back_node_id == rank_0_node.node_id
        ):
            return connection.send_back_multiaddr.ip_address

    # Fallback: use first network interface on rank-0
    if rank_0_node.node_profile and rank_0_node.node_profile.network_interfaces:
        return rank_0_node.node_profile.network_interfaces[0].ip_address

    raise ValueError(
        f"Cannot determine master address for rank-0 node {rank_0_node.node_id}"
    )


def add_instance_to_placements(
    command: CreateInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> Mapping[InstanceId, Instance]:
    # TODO: validate against topology

    return {**current_instances, command.instance.instance_id: command.instance}


def place_instance(
    command: PlaceInstance,
    topology: Topology,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    all_nodes = list(topology.list_nodes())

    logger.info("finding cycles:")
    cycles = topology.get_cycles()
    singleton_cycles = [[node] for node in all_nodes]
    candidate_cycles = list(
        filter(lambda it: len(it) >= command.min_nodes, cycles + singleton_cycles)
    )

    # Check if this is a CUDA instance request
    is_cuda_instance = command.instance_meta in (
        InstanceMeta.CudaNccl,
        InstanceMeta.CudaGloo,
    )

    if is_cuda_instance:
        # For CUDA instances, filter by GPU memory and CUDA capability
        logger.info("CUDA instance requested - applying GPU-aware filtering")

        # First, filter to cycles that have CUDA GPUs
        cycles_with_cuda = filter_cycles_by_cuda_capability(candidate_cycles)
        if not cycles_with_cuda:
            raise ValueError(
                "No cycles found with CUDA-capable GPUs. "
                "CUDA instances require NVIDIA GPUs on all participating nodes."
            )

        # Filter by GPU memory (model must fit in GPU VRAM)
        cycles_with_sufficient_memory = filter_cycles_by_gpu_memory(
            cycles_with_cuda, command.model_meta.storage_size
        )
        if not cycles_with_sufficient_memory:
            # Fall back to RAM-based filtering if no cycles have enough GPU memory
            # This allows placement on systems where model offloading might be used
            logger.warning(
                "No cycles with sufficient GPU memory, falling back to RAM filtering"
            )
            cycles_with_sufficient_memory = filter_cycles_by_memory(
                cycles_with_cuda, command.model_meta.storage_size
            )

        if not cycles_with_sufficient_memory:
            raise ValueError(
                "No CUDA-capable cycles found with sufficient memory for model"
            )

        # Rank by GPU quality (memory, NVLink support)
        cycles_with_sufficient_memory = rank_cycles_by_gpu_quality(
            cycles_with_sufficient_memory
        )
    else:
        # For non-CUDA instances (MLX, etc.), use RAM-based filtering
        cycles_with_sufficient_memory = filter_cycles_by_memory(
            candidate_cycles, command.model_meta.storage_size
        )
        if not cycles_with_sufficient_memory:
            raise ValueError("No cycles found with sufficient memory")

    if command.sharding == Sharding.Tensor:
        if not command.model_meta.supports_tensor:
            raise ValueError(
                f"Requested Tensor sharding but this model does not support tensor parallelism: {command.model_meta.model_id}"
            )
        # TODO: the condition here for tensor parallel is not correct, but it works good enough for now.
        cycles_with_sufficient_memory = [
            cycle
            for cycle in cycles_with_sufficient_memory
            if command.model_meta.hidden_size % len(cycle) == 0
        ]
        if not cycles_with_sufficient_memory:
            raise ValueError(
                f"No tensor sharding found for model with hidden_size {command.model_meta.hidden_size} candidate cycles"
            )
    if command.sharding == Sharding.Pipeline and command.model_meta.model_id == ModelId(
        "mlx-community/DeepSeek-V3.1-8bit"
    ):
        raise ValueError(
            "Pipeline parallelism is not supported for DeepSeek V3.1 (8-bit)"
        )

    smallest_cycles = get_smallest_cycles(cycles_with_sufficient_memory)

    smallest_tb_cycles = [
        cycle
        for cycle in smallest_cycles
        if topology.get_subgraph_from_nodes(cycle).is_thunderbolt_cycle(cycle)
    ]

    if smallest_tb_cycles != []:
        smallest_cycles = smallest_tb_cycles

    cycles_with_leaf_nodes: list[list[NodeInfo]] = [
        cycle
        for cycle in smallest_cycles
        if any(topology.node_is_leaf(node.node_id) for node in cycle)
    ]

    selected_cycle = max(
        cycles_with_leaf_nodes if cycles_with_leaf_nodes != [] else smallest_cycles,
        key=lambda cycle: sum(
            (
                node.node_profile.memory.ram_available
                for node in cycle
                if node.node_profile is not None
            ),
            start=Memory(),
        ),
    )

    shard_assignments = get_shard_assignments(
        command.model_meta, selected_cycle, command.sharding
    )

    cycle_digraph: Topology = topology.get_subgraph_from_nodes(selected_cycle)

    instance_id = InstanceId()
    target_instances = dict(deepcopy(current_instances))

    if len(selected_cycle) == 1:
        logger.warning(
            "You have likely selected ibv for a single node instance; falling back to MlxRing"
        )

        command.instance_meta = InstanceMeta.MlxRing

    # TODO: Single node instances
    match command.instance_meta:
        case InstanceMeta.MlxJaccl:
            mlx_ibv_devices = get_mlx_ibv_devices_matrix(
                selected_cycle,
                cycle_digraph,
            )
            mlx_jaccl_coordinators = get_mlx_jaccl_coordinators(
                selected_cycle,
                coordinator_port=random_ephemeral_port(),
                cycle_digraph=cycle_digraph,
            )
            target_instances[instance_id] = MlxJacclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                ibv_devices=mlx_ibv_devices,
                jaccl_coordinators=mlx_jaccl_coordinators,
            )
        case InstanceMeta.MlxRing:
            ephemeral_port = random_ephemeral_port()
            hosts_by_node = get_mlx_ring_hosts_by_node(
                selected_cycle=selected_cycle,
                cycle_digraph=cycle_digraph,
                ephemeral_port=ephemeral_port,
            )
            target_instances[instance_id] = MlxRingInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                hosts_by_node=hosts_by_node,
                ephemeral_port=ephemeral_port,
            )

        case InstanceMeta.CudaNccl:
            # Get device IDs for each node in the cycle
            device_ids_by_node = get_cuda_device_ids_for_cycle(selected_cycle)

            # For now, use first GPU on each node (device 0)
            # Future: support multi-GPU per node with tensor parallelism
            device_ids = [
                device_ids_by_node.get(node.node_id, [0])[0]
                if device_ids_by_node.get(node.node_id)
                else 0
                for node in selected_cycle
            ]

            # Get master address and port for NCCL initialization
            master_addr = get_cuda_master_addr(selected_cycle, cycle_digraph)
            master_port = random_ephemeral_port()

            # Generate NCCL unique ID for communicator setup
            nccl_unique_id = generate_nccl_unique_id()

            logger.info(
                f"Creating CUDA NCCL instance with {len(selected_cycle)} nodes, "
                f"master={master_addr}:{master_port}"
            )

            target_instances[instance_id] = CudaNcclInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                nccl_unique_id=nccl_unique_id,
                device_ids=device_ids,
                master_addr=master_addr,
                master_port=master_port,
            )

        case InstanceMeta.CudaGloo:
            # Get device IDs for each node in the cycle
            device_ids_by_node = get_cuda_device_ids_for_cycle(selected_cycle)

            # For now, use first GPU on each node (device 0)
            device_ids = [
                device_ids_by_node.get(node.node_id, [0])[0]
                if device_ids_by_node.get(node.node_id)
                else 0
                for node in selected_cycle
            ]

            # Get master address and port for Gloo initialization
            master_addr = get_cuda_master_addr(selected_cycle, cycle_digraph)
            master_port = random_ephemeral_port()

            logger.info(
                f"Creating CUDA Gloo instance with {len(selected_cycle)} nodes, "
                f"master={master_addr}:{master_port}"
            )

            target_instances[instance_id] = CudaGlooInstance(
                instance_id=instance_id,
                shard_assignments=shard_assignments,
                master_addr=master_addr,
                master_port=master_port,
                device_ids=device_ids,
            )

        case InstanceMeta.VulkanCompute:
            # TODO: Implement Vulkan compute instance placement (Phase 8)
            raise NotImplementedError(
                "Vulkan compute instance placement is not yet implemented. "
                "This will be added in Phase 8: Vulkan Backend."
            )

    return target_instances


def delete_instance(
    command: DeleteInstance,
    current_instances: Mapping[InstanceId, Instance],
) -> dict[InstanceId, Instance]:
    target_instances = dict(deepcopy(current_instances))
    if command.instance_id in target_instances:
        del target_instances[command.instance_id]
        return target_instances
    raise ValueError(f"Instance {command.instance_id} not found")


def get_transition_events(
    current_instances: Mapping[InstanceId, Instance],
    target_instances: Mapping[InstanceId, Instance],
) -> Sequence[Event]:
    events: list[Event] = []

    # find instances to create
    for instance_id, instance in target_instances.items():
        if instance_id not in current_instances:
            events.append(
                InstanceCreated(
                    instance=instance,
                )
            )

    # find instances to delete
    for instance_id in current_instances:
        if instance_id not in target_instances:
            events.append(
                InstanceDeleted(
                    instance_id=instance_id,
                )
            )

    return events

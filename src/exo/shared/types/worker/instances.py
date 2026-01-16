from enum import Enum

from pydantic import model_validator

from exo.shared.types.common import Host, Id, NodeId
from exo.shared.types.worker.runners import RunnerId, ShardAssignments, ShardMetadata
from exo.utils.pydantic_ext import CamelCaseModel, TaggedModel


class InstanceId(Id):
    pass


class InstanceMeta(str, Enum):
    """Metadata enum identifying the type of inference backend instance."""

    # Apple MLX backends
    MlxRing = "MlxRing"  # MLX with Ring backend (TCP/Ethernet)
    MlxJaccl = "MlxJaccl"  # MLX with JACCL backend (RDMA/Thunderbolt 5)

    # NVIDIA CUDA backends
    CudaNccl = "CudaNccl"  # CUDA with NCCL (optimized GPU-to-GPU communication)
    CudaGloo = "CudaGloo"  # CUDA with Gloo (CPU-based fallback for distributed)

    # Vulkan compute backend (future)
    VulkanCompute = "VulkanCompute"  # Vulkan compute for cross-vendor GPU support


class BaseInstance(TaggedModel):
    instance_id: InstanceId
    shard_assignments: ShardAssignments

    def shard(self, runner_id: RunnerId) -> ShardMetadata | None:
        return self.shard_assignments.runner_to_shard.get(runner_id, None)


# =============================================================================
# MLX Instance Types (Apple Silicon)
# =============================================================================


class MlxRingInstance(BaseInstance):
    """
    MLX Ring backend instance for TCP/Ethernet communication.

    Used for distributed inference on Apple Silicon devices connected via
    standard networking (Ethernet, WiFi).
    """

    hosts_by_node: dict[NodeId, list[Host]]
    ephemeral_port: int


class MlxJacclInstance(BaseInstance):
    """
    MLX JACCL backend instance for RDMA/Thunderbolt 5 communication.

    Used for high-speed distributed inference on Apple Silicon devices
    connected via Thunderbolt 5 with RDMA support.
    """

    ibv_devices: list[list[str | None]]
    jaccl_coordinators: dict[NodeId, str]


# =============================================================================
# CUDA Instance Types (NVIDIA GPUs)
# =============================================================================


class CudaNcclInstance(BaseInstance):
    """
    CUDA instance using NCCL for distributed communication.

    NCCL (NVIDIA Collective Communications Library) provides optimized
    GPU-to-GPU communication primitives for multi-GPU and multi-node
    distributed training and inference.

    Attributes:
        nccl_unique_id: Unique identifier for the NCCL communicator group.
            Generated on the rank-0 process and shared with other ranks.
        device_ids: List of CUDA device indices assigned to this instance.
            For single-GPU nodes, typically [0]. For multi-GPU nodes, could be
            [0, 1, 2, 3] for a 4-GPU configuration.
        master_addr: IP address of the rank-0 node for NCCL initialization.
        master_port: Port number for NCCL initialization on the master node.
    """

    nccl_unique_id: str
    device_ids: list[int]
    master_addr: str
    master_port: int


class CudaGlooInstance(BaseInstance):
    """
    CUDA instance using Gloo for distributed communication.

    Gloo is a CPU-based collective communications library that works as a
    fallback when NCCL is not available or for CPU-only operations. It's
    useful for:
    - Nodes without direct GPU-to-GPU communication (e.g., different subnets)
    - Mixed CPU/GPU workloads
    - Debugging distributed setups

    Attributes:
        master_addr: IP address of the rank-0 node for Gloo initialization.
        master_port: Port number for Gloo initialization on the master node.
        device_ids: List of CUDA device indices assigned to this instance.
    """

    master_addr: str
    master_port: int
    device_ids: list[int]


# =============================================================================
# Vulkan Instance Types (Cross-vendor GPU support)
# =============================================================================


class VulkanComputeInstance(BaseInstance):
    """
    Vulkan compute instance for cross-vendor GPU support.

    Vulkan provides a unified API for GPU compute across NVIDIA, AMD, and Intel
    GPUs. This enables distributed inference on heterogeneous GPU clusters.

    Attributes:
        device_index: Vulkan physical device index on this node.
        device_name: Human-readable name of the Vulkan device (e.g., "NVIDIA RTX 4090").
        memory_budget: Memory budget in bytes for this instance on the device.
        master_addr: IP address of the rank-0 node for coordination.
        master_port: Port number for coordination on the master node.
    """

    device_index: int
    device_name: str
    memory_budget: int
    master_addr: str
    master_port: int


# =============================================================================
# Instance Type Union
# =============================================================================

# Union of all supported instance types for type checking
# TODO: Single node instance
Instance = (
    MlxRingInstance
    | MlxJacclInstance
    | CudaNcclInstance
    | CudaGlooInstance
    | VulkanComputeInstance
)


class BoundInstance(CamelCaseModel):
    instance: Instance
    bound_runner_id: RunnerId
    bound_node_id: NodeId

    @property
    def bound_shard(self) -> ShardMetadata:
        shard = self.instance.shard(self.bound_runner_id)
        assert shard is not None
        return shard

    @model_validator(mode="after")
    def validate_shard_exists(self) -> "BoundInstance":
        assert (
            self.bound_runner_id in self.instance.shard_assignments.runner_to_shard
        ), (
            "Bound Instance must be constructed with a runner_id that is in the instances assigned shards"
        )
        return self

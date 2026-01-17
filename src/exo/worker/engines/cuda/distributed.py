# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnusedImport=false
"""
Distributed communication groups for CUDA/PyTorch backend.

This module provides distributed communication implementations for multi-GPU and
multi-node inference using PyTorch's distributed backend. Two backends are supported:

- NCCL: NVIDIA Collective Communications Library for optimized GPU-to-GPU communication.
  Best for NVIDIA GPUs on Linux with direct GPU interconnects (NVLink, PCIe).

- Gloo: CPU-based collective communications for broader compatibility.
  Best for Windows, mixed CPU/GPU workloads, or debugging.

Usage:
    # NCCL backend (Linux with NVIDIA GPUs)
    group = NcclDistributedGroup(bound_instance)

    # Gloo backend (Windows or CPU fallback)
    group = GlooDistributedGroup(bound_instance)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger

from exo.shared.types.worker.instances import (
    BoundInstance,
    CudaGlooInstance,
    CudaNcclInstance,
)
from exo.worker.engines.base import DistributedGroup

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist


class NcclDistributedGroup(DistributedGroup):
    """
    NCCL-based distributed communication group.

    NCCL (NVIDIA Collective Communications Library) provides optimized collective
    operations for NVIDIA GPUs. It's the preferred backend for multi-GPU training
    and inference on Linux systems.

    Features:
        - Optimized GPU-to-GPU communication
        - Supports NVLink, NVSwitch, PCIe, InfiniBand
        - Automatic topology detection and optimization
        - Asynchronous operations for overlapping compute and communication

    Requirements:
        - Linux (NCCL has limited Windows support)
        - NVIDIA GPU with CUDA support
        - PyTorch with CUDA and NCCL support

    Attributes:
        _rank: The rank of this process in the distributed group.
        _world_size: The total number of processes in the group.
        _process_group: The PyTorch distributed process group.
    """

    def __init__(self, bound_instance: BoundInstance) -> None:
        """
        Initialize the NCCL distributed group.

        Args:
            bound_instance: The bound instance containing NCCL configuration
                including nccl_unique_id, device_ids, master_addr, and master_port.

        Raises:
            TypeError: If the instance is not a CudaNcclInstance.
            RuntimeError: If NCCL initialization fails.
        """
        import torch
        import torch.distributed as dist

        instance = bound_instance.instance
        if not isinstance(instance, CudaNcclInstance):
            raise TypeError(
                f"NcclDistributedGroup requires CudaNcclInstance, got {type(instance).__name__}"
            )

        self._rank = bound_instance.bound_shard.device_rank
        shard_assignments = instance.shard_assignments
        self._world_size = len(shard_assignments.runner_to_shard)

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = instance.master_addr
        os.environ["MASTER_PORT"] = str(instance.master_port)
        os.environ["NCCL_UNIQUE_ID"] = instance.nccl_unique_id

        # Set CUDA device
        if instance.device_ids:
            device_id = instance.device_ids[0]  # Use first device for this rank
            torch.cuda.set_device(device_id)
            logger.info(f"Rank {self._rank} using CUDA device {device_id}")

        # Initialize the process group
        logger.info(
            f"Initializing NCCL process group: rank={self._rank}, "
            f"world_size={self._world_size}, master={instance.master_addr}:{instance.master_port}"
        )

        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=self._world_size,
            rank=self._rank,
        )

        self._process_group = dist.group.WORLD
        logger.info(f"NCCL distributed group initialized for rank {self._rank}")

    def rank(self) -> int:
        """Return the rank of this process in the distributed group."""
        return self._rank

    def size(self) -> int:
        """Return the total size of the distributed group."""
        return self._world_size

    def cleanup(self) -> None:
        """Clean up the distributed group and release resources."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"NCCL distributed group destroyed for rank {self._rank}")


class GlooDistributedGroup(DistributedGroup):
    """
    Gloo-based distributed communication group.

    Gloo is a CPU-based collective communications library that provides broader
    platform compatibility than NCCL. It's useful for:

    - Windows systems where NCCL support is limited
    - Mixed CPU/GPU workloads
    - Debugging and development
    - Systems without direct GPU interconnects

    Features:
        - Cross-platform support (Linux, Windows, macOS)
        - TCP/IP-based communication
        - Works without specialized GPU interconnects

    Limitations:
        - Slower than NCCL for GPU-to-GPU communication
        - Data passes through CPU memory

    Attributes:
        _rank: The rank of this process in the distributed group.
        _world_size: The total number of processes in the group.
        _process_group: The PyTorch distributed process group.
    """

    def __init__(self, bound_instance: BoundInstance) -> None:
        """
        Initialize the Gloo distributed group.

        Args:
            bound_instance: The bound instance containing Gloo configuration
                including master_addr, master_port, and device_ids.

        Raises:
            TypeError: If the instance is not a CudaGlooInstance.
            RuntimeError: If Gloo initialization fails.
        """
        import torch
        import torch.distributed as dist

        instance = bound_instance.instance
        if not isinstance(instance, CudaGlooInstance):
            raise TypeError(
                f"GlooDistributedGroup requires CudaGlooInstance, got {type(instance).__name__}"
            )

        self._rank = bound_instance.bound_shard.device_rank
        shard_assignments = instance.shard_assignments
        self._world_size = len(shard_assignments.runner_to_shard)

        # Set environment variables for torch.distributed
        os.environ["MASTER_ADDR"] = instance.master_addr
        os.environ["MASTER_PORT"] = str(instance.master_port)

        # Set CUDA device if available
        if instance.device_ids and torch.cuda.is_available():
            device_id = instance.device_ids[0]
            torch.cuda.set_device(device_id)
            logger.info(f"Rank {self._rank} using CUDA device {device_id}")

        # Initialize the process group
        logger.info(
            f"Initializing Gloo process group: rank={self._rank}, "
            f"world_size={self._world_size}, master={instance.master_addr}:{instance.master_port}"
        )

        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            world_size=self._world_size,
            rank=self._rank,
        )

        self._process_group = dist.group.WORLD
        logger.info(f"Gloo distributed group initialized for rank {self._rank}")

    def rank(self) -> int:
        """Return the rank of this process in the distributed group."""
        return self._rank

    def size(self) -> int:
        """Return the total size of the distributed group."""
        return self._world_size

    def cleanup(self) -> None:
        """Clean up the distributed group and release resources."""
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info(f"Gloo distributed group destroyed for rank {self._rank}")

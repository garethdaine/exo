"""
MLX inference engine implementation.

This module provides the MlxEngine class which implements the InferenceEngine
protocol for Apple Silicon devices using the MLX framework.

The MlxEngine wraps existing MLX functionality to provide a consistent interface
that can be used alongside other backends (CUDA, Vulkan) in a heterogeneous cluster.
"""

from collections.abc import Generator
from typing import cast, final

import mlx.core as mx
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.base import (
    DistributedGroup,
    EngineCapabilities,
    EngineInfo,
    InferenceEngine,
    TimeoutCallback,
)
from exo.worker.engines.mlx import Model
from exo.worker.engines.mlx.generator.generate import mlx_generate, warmup_inference
from exo.worker.engines.mlx.utils_mlx import (
    initialize_mlx,
    load_mlx_items,
    mlx_cleanup,
)

# Type alias for MLX distributed group
MlxDistributedGroup = mx.distributed.Group


@final
class MlxDistributedGroupWrapper:
    """
    Wrapper around MLX's distributed Group to implement the DistributedGroup protocol.

    This class provides a consistent interface for the runner to interact with
    distributed communication, regardless of whether MLX Ring or JACCL backend is used.
    """

    def __init__(self, mlx_group: MlxDistributedGroup) -> None:
        """
        Initialize the wrapper with an MLX distributed group.

        Args:
            mlx_group: The underlying MLX distributed group.
        """
        self._group = mlx_group

    def rank(self) -> int:
        """
        Return the rank of this process in the distributed group.

        Returns:
            Integer rank (0-indexed), where rank 0 is the coordinator.
        """
        return int(self._group.rank())

    def size(self) -> int:
        """
        Return the total size of the distributed group.

        Returns:
            Number of processes in the group.
        """
        return int(self._group.size())

    @property
    def underlying_group(self) -> MlxDistributedGroup:
        """
        Access the underlying MLX distributed group.

        This property allows engine-internal code to access the native MLX
        group when needed for operations that require the concrete type.

        Returns:
            The underlying mx.distributed.Group instance.
        """
        return self._group


@final
class MlxEngine:
    """
    MLX inference engine for Apple Silicon devices.

    This class implements the InferenceEngine protocol to provide a consistent
    interface for distributed inference using the MLX framework. It supports
    both tensor parallelism (via Ring or JACCL backends) and pipeline parallelism.

    The engine wraps the existing MLX utility functions to maintain backward
    compatibility while providing the abstraction layer needed for multi-backend
    support.

    Usage:
        engine = MlxEngine()
        group = engine.initialize_distributed(bound_instance)
        model, tokenizer = engine.load_model(bound_instance, group)
        engine.warmup(model, tokenizer)

        for response in engine.generate(model, tokenizer, task):
            # Process generation response
            pass

        engine.cleanup(model, tokenizer, group)
    """

    def __init__(self) -> None:
        """Initialize the MLX engine."""
        self._info = EngineInfo(
            name="mlx",
            platform="darwin",
            capabilities=EngineCapabilities(
                supports_tensor_parallelism=True,
                supports_pipeline_parallelism=True,
                supports_quantized_kv_cache=True,
                supports_streaming=True,
                max_sequence_length=None,  # Model-dependent
                supported_dtypes=["float16", "bfloat16", "float32"],
            ),
        )

    @property
    def info(self) -> EngineInfo:
        """
        Return metadata about this engine.

        Returns:
            EngineInfo containing name, platform, and capabilities.
        """
        return self._info

    def initialize_distributed(
        self,
        bound_instance: BoundInstance,
    ) -> DistributedGroup:
        """
        Initialize the MLX distributed communication backend.

        This method sets up the MLX distributed group for multi-node or multi-GPU
        inference. It supports both Ring (TCP/Ethernet) and JACCL (RDMA/Thunderbolt 5)
        backends based on the instance type.

        Args:
            bound_instance: The bound instance containing shard assignments
                and node-specific configuration.

        Returns:
            A MlxDistributedGroupWrapper implementing the DistributedGroup protocol.

        Raises:
            RuntimeError: If distributed initialization fails.
            AssertionError: If called for a single-node instance.
        """
        mlx_group = initialize_mlx(bound_instance)
        return MlxDistributedGroupWrapper(mlx_group)

    def load_model(
        self,
        bound_instance: BoundInstance,
        group: DistributedGroup | None,
        on_timeout: TimeoutCallback | None = None,
    ) -> tuple[Model, TokenizerWrapper]:
        """
        Load and optionally shard the MLX model across devices.

        This method loads the model weights using mlx-lm, applies tensor or
        pipeline parallelism sharding if a distributed group is provided,
        and returns the model and tokenizer.

        Args:
            bound_instance: The bound instance containing model metadata,
                shard assignments, and configuration.
            group: The distributed group for multi-device sharding, or None
                for single-device inference.
            on_timeout: Optional callback to invoke if model loading times out.
                This allows the runner to send a failure event before termination.

        Returns:
            A tuple of (model, tokenizer) where:
                - model: The loaded MLX model (mlx.nn.Module subclass)
                - tokenizer: The loaded TokenizerWrapper

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
            ModelLoadingTimeoutError: If model loading exceeds the timeout.
        """
        # Extract the underlying MLX group if wrapped
        mlx_group: MlxDistributedGroup | None = None
        if group is not None:
            if isinstance(group, MlxDistributedGroupWrapper):
                mlx_group = group.underlying_group
            else:
                # Allow passing native MLX group for backward compatibility
                mlx_group = cast(MlxDistributedGroup, group)

        return load_mlx_items(bound_instance, mlx_group, on_timeout)

    def warmup(
        self,
        model: object,
        tokenizer: object,
    ) -> int:
        """
        Run warmup inference to prime GPU caches and Metal compilation.

        Warmup ensures the first real inference request doesn't pay the cost
        of Metal shader compilation, memory allocation, and cache priming.

        Args:
            model: The loaded MLX model.
            tokenizer: The loaded TokenizerWrapper.

        Returns:
            The number of tokens generated during warmup.
        """
        return warmup_inference(
            model=cast(Model, model),
            tokenizer=cast(TokenizerWrapper, tokenizer),
        )

    def generate(
        self,
        model: object,
        tokenizer: object,
        task: ChatCompletionTaskParams,
    ) -> Generator[GenerationResponse, None, None]:
        """
        Generate tokens from a chat completion task using MLX.

        This method yields GenerationResponse objects for each generated token,
        allowing streaming output to the client. The generation uses the MLX
        stream_generate function with the configured sampling parameters.

        Args:
            model: The loaded MLX model.
            tokenizer: The loaded TokenizerWrapper.
            task: The chat completion task parameters including messages,
                sampling parameters, and generation limits.

        Yields:
            GenerationResponse objects containing the generated token text,
            token ID, and optional finish reason and stats.

        Raises:
            RuntimeError: If generation fails due to an MLX error.
        """
        yield from mlx_generate(
            model=cast(Model, model),
            tokenizer=cast(TokenizerWrapper, tokenizer),
            task=task,
        )

    def cleanup(
        self,
        model: object | None,
        tokenizer: object | None,
        group: DistributedGroup | None,
    ) -> None:
        """
        Clean up MLX resources.

        This method frees Metal device memory, clears MLX caches, and
        performs garbage collection.

        Args:
            model: The loaded model to clean up, or None if not loaded.
            tokenizer: The tokenizer to clean up, or None if not loaded.
            group: The distributed group to clean up, or None if not initialized.
        """
        # Extract underlying MLX group if wrapped
        mlx_group: MlxDistributedGroup | None = None
        if group is not None:
            if isinstance(group, MlxDistributedGroupWrapper):
                mlx_group = group.underlying_group
            else:
                mlx_group = cast(MlxDistributedGroup, group)

        mlx_cleanup(
            cast(Model | None, model),
            cast(TokenizerWrapper | None, tokenizer),
            mlx_group,
        )


# Module-level engine instance for convenience
# This allows simple usage: from exo.worker.engines.mlx.engine import mlx_engine
mlx_engine = MlxEngine()


def get_mlx_engine() -> InferenceEngine:
    """
    Get the MLX inference engine instance.

    Returns:
        The MlxEngine singleton implementing InferenceEngine protocol.
    """
    return mlx_engine

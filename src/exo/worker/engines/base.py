"""
Base protocols for inference engine abstraction.

This module defines the core abstractions that all inference backends (MLX, CUDA/PyTorch,
Vulkan) must implement. The abstraction layer enables Exo to support heterogeneous clusters
with different hardware accelerators.

Design Principles:
    - Protocol-based design for structural typing (no inheritance required)
    - Type-safe interfaces with strict typing discipline
    - Engine-agnostic types for cross-backend communication
    - Pure functions where possible, with side effects isolated to well-defined methods
"""

from abc import abstractmethod
from collections.abc import Callable, Generator
from typing import Any, Protocol, runtime_checkable

from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import BoundInstance
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.shards import ShardMetadata

# Type alias for timeout callbacks used during model loading
# Defined early to avoid forward reference issues
TimeoutCallback = Callable[[], None]


@runtime_checkable
class Tokenizer(Protocol):
    """
    Protocol for tokenizer implementations across different backends.

    A tokenizer is responsible for converting text to token IDs and vice versa,
    as well as applying chat templates for conversation formatting.

    All backends (MLX, PyTorch, etc.) must provide a tokenizer that implements
    this interface to ensure consistent behavior across the cluster.
    """

    @property
    def eos_token_ids(self) -> list[int]:
        """
        Return the end-of-sequence token IDs for this tokenizer.

        Returns:
            List of token IDs that signal generation completion.
        """
        ...

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text into a list of token IDs.

        Args:
            text: The input text to tokenize.
            add_special_tokens: Whether to add special tokens (BOS, EOS, etc.).

        Returns:
            List of integer token IDs.
        """
        ...

    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        tools: list[Any] | None = None,
    ) -> str:
        """
        Apply a chat template to format conversation messages.

        Args:
            messages_dicts: List of message dictionaries with role and content.
            tokenize: Whether to return token IDs instead of string.
            add_generation_prompt: Whether to add the assistant prompt prefix.
            tools: Optional list of tool definitions for function calling.

        Returns:
            Formatted prompt string (or token IDs if tokenize=True).
        """
        ...


@runtime_checkable
class Detokenizer(Protocol):
    """
    Protocol for streaming detokenization.

    A detokenizer converts token IDs back to text in a streaming fashion,
    handling partial UTF-8 sequences and special token boundaries.
    """

    def reset(self) -> None:
        """Reset the detokenizer state for a new generation."""
        ...

    def add_token(self, token: int) -> None:
        """
        Add a token to the detokenization buffer.

        Args:
            token: The token ID to add.
        """
        ...

    def finalize(self) -> None:
        """Finalize the detokenization, flushing any remaining tokens."""
        ...

    @property
    def last_segment(self) -> str:
        """
        Return the last decoded text segment.

        Returns:
            The most recently decoded text string.
        """
        ...


@runtime_checkable
class DistributedGroup(Protocol):
    """
    Protocol for distributed communication groups across backends.

    A distributed group coordinates communication between nodes/devices in
    a distributed inference setup. Different backends use different primitives:
    - MLX: Ring or JACCL backends
    - PyTorch/CUDA: NCCL or Gloo backends
    - Vulkan: Custom compute shader communication

    The protocol abstracts these differences to enable cross-backend operation.
    """

    def rank(self) -> int:
        """
        Return the rank of this process in the distributed group.

        Returns:
            Integer rank (0-indexed), where rank 0 is typically the coordinator.
        """
        ...

    def size(self) -> int:
        """
        Return the total size of the distributed group.

        Returns:
            Number of processes in the group.
        """
        ...


# Note: The InferenceEngine protocol uses `object` for model and tokenizer types
# because different backends have different concrete types (mlx.nn.Module vs torch.nn.Module).
# This is intentional - the protocol defines the interface contract, and each engine
# implementation knows its own concrete types. Using `object` is safer than `Any`
# as it still requires explicit type narrowing in implementations.


@runtime_checkable
class InferenceEngine(Protocol):
    """
    Protocol for inference engine implementations.

    An inference engine is responsible for:
    1. Initializing distributed communication (if multi-node/multi-GPU)
    2. Loading and optionally sharding models across devices
    3. Running inference and generating tokens
    4. Cleaning up resources when done

    Each backend (MLX, CUDA/PyTorch, Vulkan) implements this protocol
    to provide a consistent interface to the runner.

    The engine methods are called in order during the runner lifecycle:
        1. initialize_distributed() - Set up distributed backend
        2. load_model() - Load and shard the model
        3. warmup() - Run warmup inference to prime caches
        4. generate() - Generate tokens (may be called multiple times)
        5. cleanup() - Free resources

    Error Handling:
        Methods may raise exceptions which should be caught by the runner
        and reported via the event system. The runner is responsible for
        transitioning to the appropriate failure state.
    """

    @abstractmethod
    def initialize_distributed(
        self,
        bound_instance: BoundInstance,
    ) -> DistributedGroup:
        """
        Initialize the distributed communication backend.

        This method sets up the distributed group for multi-node or multi-GPU
        inference. It handles backend-specific initialization (ring topology,
        NCCL communicators, etc.).

        Args:
            bound_instance: The bound instance containing shard assignments
                and node-specific configuration.

        Returns:
            A DistributedGroup instance for coordinating communication.

        Raises:
            RuntimeError: If distributed initialization fails.
            TimeoutError: If peer connection times out.
        """
        ...

    @abstractmethod
    def load_model(
        self,
        bound_instance: BoundInstance,
        group: DistributedGroup | None,
        on_timeout: TimeoutCallback | None = None,
    ) -> tuple[object, object]:
        """
        Load and optionally shard the model across devices.

        This method loads the model weights from storage, applies any necessary
        sharding for distributed inference, and returns the model and tokenizer.

        Args:
            bound_instance: The bound instance containing model metadata,
                shard assignments, and configuration.
            group: The distributed group for multi-device sharding, or None
                for single-device inference.
            on_timeout: Optional callback to invoke if model loading times out.
                This allows the runner to send a failure event before termination.

        Returns:
            A tuple of (model, tokenizer) where:
                - model: The loaded model (engine-specific type)
                - tokenizer: The loaded tokenizer implementing Tokenizer protocol

        Raises:
            FileNotFoundError: If model files are not found.
            RuntimeError: If model loading fails.
            TimeoutError: If model loading exceeds the timeout.
        """
        ...

    @abstractmethod
    def warmup(
        self,
        model: object,
        tokenizer: object,
    ) -> int:
        """
        Run warmup inference to prime GPU caches and JIT compilation.

        Warmup ensures the first real inference request doesn't pay the cost
        of GPU kernel compilation, memory allocation, and cache priming.

        Args:
            model: The loaded model (engine-specific type).
            tokenizer: The loaded tokenizer.

        Returns:
            The number of tokens generated during warmup.
        """
        ...

    @abstractmethod
    def generate(
        self,
        model: object,
        tokenizer: object,
        task: ChatCompletionTaskParams,
    ) -> Generator[GenerationResponse, None, None]:
        """
        Generate tokens from a chat completion task.

        This method yields GenerationResponse objects for each generated token,
        allowing streaming output to the client. The generation continues until:
        - A stop condition is met (EOS token, stop sequence)
        - The maximum token limit is reached
        - An error occurs

        Args:
            model: The loaded model (engine-specific type).
            tokenizer: The loaded tokenizer.
            task: The chat completion task parameters including messages,
                sampling parameters, and generation limits.

        Yields:
            GenerationResponse objects containing the generated token text,
            token ID, and optional finish reason and stats.

        Raises:
            RuntimeError: If generation fails due to a backend error.
        """
        ...

    @abstractmethod
    def cleanup(
        self,
        model: object | None,
        tokenizer: object | None,
        group: DistributedGroup | None,
    ) -> None:
        """
        Clean up resources used by the engine.

        This method frees GPU memory, closes communication channels, and
        performs any other necessary cleanup.

        Args:
            model: The loaded model to clean up, or None if not loaded.
            tokenizer: The tokenizer to clean up, or None if not loaded.
            group: The distributed group to clean up, or None if not initialized.
        """
        ...


class EngineCapabilities:
    """
    Describes the capabilities of an inference engine.

    This class allows engines to declare what features they support,
    enabling the runner and master to make informed placement decisions.
    """

    def __init__(
        self,
        *,
        supports_tensor_parallelism: bool = False,
        supports_pipeline_parallelism: bool = False,
        supports_quantized_kv_cache: bool = False,
        supports_streaming: bool = True,
        max_sequence_length: int | None = None,
        supported_dtypes: list[str] | None = None,
    ) -> None:
        """
        Initialize engine capabilities.

        Args:
            supports_tensor_parallelism: Whether the engine can shard layers
                across multiple devices.
            supports_pipeline_parallelism: Whether the engine can distribute
                layers across multiple devices in a pipeline.
            supports_quantized_kv_cache: Whether the engine supports
                quantized KV cache for memory efficiency.
            supports_streaming: Whether the engine supports token-by-token
                streaming output.
            max_sequence_length: Maximum supported sequence length, or None
                for no limit.
            supported_dtypes: List of supported data types (e.g., ["float16",
                "bfloat16", "float32"]).
        """
        self.supports_tensor_parallelism = supports_tensor_parallelism
        self.supports_pipeline_parallelism = supports_pipeline_parallelism
        self.supports_quantized_kv_cache = supports_quantized_kv_cache
        self.supports_streaming = supports_streaming
        self.max_sequence_length = max_sequence_length
        self.supported_dtypes = supported_dtypes or ["float16"]


class EngineInfo:
    """
    Metadata about an inference engine implementation.

    This class provides information for engine discovery and selection,
    including the engine name, supported platforms, and capabilities.
    """

    def __init__(
        self,
        *,
        name: str,
        platform: str,
        capabilities: EngineCapabilities,
        version: str = "1.0.0",
    ) -> None:
        """
        Initialize engine info.

        Args:
            name: The engine name (e.g., "mlx", "cuda", "vulkan").
            platform: The platform this engine supports (e.g., "darwin",
                "linux", "windows").
            capabilities: The engine's capabilities.
            version: The engine version string.
        """
        self.name = name
        self.platform = platform
        self.capabilities = capabilities
        self.version = version


def get_engine_for_shard_metadata(shard_metadata: ShardMetadata) -> str:
    """
    Determine the appropriate engine type for a given shard metadata.

    This function examines the shard metadata and returns the engine name
    that should be used for inference. Currently this is a placeholder that
    always returns "mlx", but will be extended to support CUDA and Vulkan.

    Args:
        shard_metadata: The shard metadata containing model and device info.

    Returns:
        The engine name string (e.g., "mlx", "cuda", "vulkan").
    """
    # TODO: Implement engine selection logic based on:
    # - Platform detection (macOS -> MLX, Linux/Windows -> CUDA)
    # - Instance type (MlxRingInstance -> MLX, CudaNcclInstance -> CUDA)
    # - Model requirements (some models may only work with certain engines)
    _ = shard_metadata  # Suppress unused variable warning
    return "mlx"


def get_engine_name_for_instance(instance: object) -> str:
    """
    Determine the appropriate engine name based on instance type.

    This function examines the instance type and returns the engine name
    that should be used for inference.

    Args:
        instance: The instance object (MlxRingInstance, CudaNcclInstance, etc.).

    Returns:
        The engine name string ("mlx", "cuda", or "vulkan").

    Raises:
        ValueError: If the instance type is not recognized.
    """
    # Import here to avoid circular imports
    from exo.shared.types.worker.instances import (
        CudaGlooInstance,
        CudaNcclInstance,
        MlxJacclInstance,
        MlxRingInstance,
        VulkanComputeInstance,
    )

    if isinstance(instance, (MlxRingInstance, MlxJacclInstance)):
        return "mlx"
    elif isinstance(instance, (CudaNcclInstance, CudaGlooInstance)):
        return "cuda"
    elif isinstance(instance, VulkanComputeInstance):
        return "vulkan"
    else:
        raise ValueError(f"Unknown instance type: {type(instance).__name__}")

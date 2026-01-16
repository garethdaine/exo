"""
Inference engine abstractions and implementations.

This package provides the infrastructure for multiple inference backends:
- base.py: Protocol definitions for engine abstraction
- registry.py: Engine discovery and instantiation
- mlx/: Apple MLX backend for Apple Silicon
- (future) cuda/: NVIDIA CUDA backend for Linux/Windows
- (future) vulkan/: Vulkan backend for cross-vendor GPU support

Usage:
    from exo.worker.engines import get_engine, InferenceEngine

    # Get an engine by name
    engine = get_engine("mlx")

    # Or use the protocol for type hints
    def run_inference(engine: InferenceEngine) -> None:
        ...
"""

from exo.worker.engines.base import (
    DistributedGroup,
    Detokenizer,
    EngineCapabilities,
    EngineInfo,
    InferenceEngine,
    TimeoutCallback,
    Tokenizer,
    get_engine_for_shard_metadata,
)
from exo.worker.engines.registry import (
    EngineNotAvailableError,
    EngineNotFoundError,
    get_default_engine_for_platform,
    get_engine,
    is_engine_available,
    list_available_engines,
    register_engine,
    register_engine_factory,
)
from exo.worker.engines.serialization import (
    FrameworkType,
    SerializedTensor,
    TensorDtype,
    TensorMetadata,
    TensorSerializer,
    TensorSerializerProtocol,
    tensor_serializer,
)

__all__ = [
    # Base protocols and types
    "DistributedGroup",
    "Detokenizer",
    "EngineCapabilities",
    "EngineInfo",
    "InferenceEngine",
    "TimeoutCallback",
    "Tokenizer",
    "get_engine_for_shard_metadata",
    # Registry
    "EngineNotAvailableError",
    "EngineNotFoundError",
    "get_default_engine_for_platform",
    "get_engine",
    "is_engine_available",
    "list_available_engines",
    "register_engine",
    "register_engine_factory",
    # Serialization
    "FrameworkType",
    "SerializedTensor",
    "TensorDtype",
    "TensorMetadata",
    "TensorSerializer",
    "TensorSerializerProtocol",
    "tensor_serializer",
]

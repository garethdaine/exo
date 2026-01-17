# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
# pyright: reportAny=false
"""
CUDA inference engine implementation.

This module provides the CudaEngine class that implements the InferenceEngine
protocol for NVIDIA CUDA GPUs using PyTorch.

The engine supports:
- Single-GPU inference
- Multi-GPU tensor parallelism via NCCL
- Multi-node distributed inference
- HuggingFace Transformers model loading
- Streaming token generation

Usage:
    from exo.worker.engines.cuda import CudaEngine

    engine = CudaEngine()
    group = engine.initialize_distributed(bound_instance)
    model, tokenizer = engine.load_model(bound_instance, group)

    for response in engine.generate(model, tokenizer, task):
        print(response.text, end="", flush=True)
"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    CudaGlooInstance,
    CudaNcclInstance,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.worker.engines.base import (
    DistributedGroup,
    EngineCapabilities,
    EngineInfo,
    InferenceEngine,
    TimeoutCallback,
)
from exo.worker.engines.cuda.distributed import (
    GlooDistributedGroup,
    NcclDistributedGroup,
)
from exo.worker.engines.cuda.generator.generate import (
    cuda_generate,
    warmup_cuda_inference,
)
from exo.worker.engines.cuda.utils_cuda import (
    clear_cuda_cache,
    get_cuda_device_info,
    get_optimal_dtype,
    initialize_cuda_device,
    is_cuda_available,
)

if TYPE_CHECKING:
    pass


class CudaEngine(InferenceEngine):
    """
    CUDA/PyTorch inference engine.

    This engine implements the InferenceEngine protocol for NVIDIA GPUs,
    providing distributed inference capabilities using PyTorch and NCCL/Gloo.

    Features:
        - HuggingFace Transformers model loading
        - NCCL distributed communication for multi-GPU
        - Gloo fallback for Windows and debugging
        - Automatic device selection and memory management
        - Support for bfloat16/float16 inference

    Example:
        engine = CudaEngine()

        # Initialize distributed group
        group = engine.initialize_distributed(bound_instance)

        # Load model
        model, tokenizer = engine.load_model(bound_instance, group)

        # Warmup
        engine.warmup(model, tokenizer)

        # Generate
        for response in engine.generate(model, tokenizer, task):
            print(response.text, end="", flush=True)

        # Cleanup
        engine.cleanup(model, tokenizer, group)
    """

    def __init__(self) -> None:
        """Initialize the CUDA engine."""
        if not is_cuda_available():
            raise RuntimeError(
                "CUDA is not available. Ensure PyTorch is installed with CUDA support."
            )

        # Log device info
        device_info = get_cuda_device_info(0)
        logger.info(
            f"CudaEngine initialized: {device_info.name} "
            f"({device_info.total_memory_gb:.2f} GB)"
        )

    @staticmethod
    def get_info() -> EngineInfo:
        """Get information about this engine."""
        return EngineInfo(
            name="cuda",
            platform="linux",  # Also supports windows
            capabilities=EngineCapabilities(
                supports_tensor_parallelism=True,
                supports_pipeline_parallelism=True,
                supports_quantized_kv_cache=False,  # TODO: implement
                supports_streaming=True,
                max_sequence_length=None,
                supported_dtypes=["float16", "bfloat16", "float32"],
            ),
        )

    def initialize_distributed(
        self,
        bound_instance: BoundInstance,
    ) -> DistributedGroup:
        """
        Initialize the distributed communication backend.

        This method creates either an NCCL or Gloo distributed group
        based on the instance type.

        Args:
            bound_instance: The bound instance with distributed configuration.

        Returns:
            A DistributedGroup for coordinating multi-GPU communication.

        Raises:
            TypeError: If the instance type is not a CUDA instance.
        """
        instance = bound_instance.instance

        if isinstance(instance, CudaNcclInstance):
            logger.info("Initializing NCCL distributed group")
            return NcclDistributedGroup(bound_instance)

        elif isinstance(instance, CudaGlooInstance):
            logger.info("Initializing Gloo distributed group")
            return GlooDistributedGroup(bound_instance)

        else:
            raise TypeError(
                f"CudaEngine requires CudaNcclInstance or CudaGlooInstance, "
                f"got {type(instance).__name__}"
            )

    def load_model(
        self,
        bound_instance: BoundInstance,
        group: DistributedGroup | None,
        on_timeout: TimeoutCallback | None = None,
    ) -> tuple[object, object]:
        """
        Load a model for CUDA inference.

        This method loads a model from HuggingFace Hub or local storage,
        configures it for inference, and optionally applies tensor parallelism.

        Args:
            bound_instance: The bound instance with model and shard metadata.
            group: The distributed group for multi-GPU sharding.
            on_timeout: Optional callback if loading times out.

        Returns:
            A tuple of (model, tokenizer).

        Raises:
            RuntimeError: If model loading fails.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        shard_metadata = bound_instance.bound_shard
        model_id = shard_metadata.model_meta.model_id

        logger.info(f"Loading CUDA model: {model_id}")

        # Determine device
        instance = bound_instance.instance
        device_id = 0
        if isinstance(instance, (CudaNcclInstance, CudaGlooInstance)) and instance.device_ids:
            device_id = instance.device_ids[0]

        device = initialize_cuda_device(device_id)
        dtype = get_optimal_dtype()

        logger.info(f"Using device: {device}, dtype: {dtype}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_id),
            trust_remote_code=True,
        )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_id),
            torch_dtype=dtype,
            device_map={"": device},
            trust_remote_code=True,
        )

        # Set to evaluation mode
        model.eval()

        logger.info(f"Model loaded: {model_id} ({dtype})")

        return model, tokenizer

    def warmup(
        self,
        model: object,
        tokenizer: object,
    ) -> int:
        """
        Warm up the CUDA inference pipeline.

        Args:
            model: The loaded PyTorch model.
            tokenizer: The loaded tokenizer.

        Returns:
            The number of tokens generated during warmup.
        """
        from transformers import PreTrainedModel

        # Type narrowing for pyright
        assert isinstance(model, PreTrainedModel)
        assert hasattr(tokenizer, "encode")

        return warmup_cuda_inference(
            model=model,
            tokenizer=cast(Any, tokenizer),
            warmup_tokens=10,
        )

    def generate(
        self,
        model: object,
        tokenizer: object,
        task: ChatCompletionTaskParams,
    ) -> Generator[GenerationResponse, None, None]:
        """
        Generate tokens using the CUDA model.

        Args:
            model: The loaded PyTorch model.
            tokenizer: The loaded tokenizer.
            task: The chat completion task parameters.

        Yields:
            GenerationResponse objects for each generated token.
        """
        from transformers import PreTrainedModel

        # Type narrowing
        assert isinstance(model, PreTrainedModel)
        assert hasattr(tokenizer, "encode")

        yield from cuda_generate(
            model=model,
            tokenizer=cast(Any, tokenizer),
            task=task,
        )

    def cleanup(
        self,
        model: object | None,
        tokenizer: object | None,
        group: DistributedGroup | None,
    ) -> None:
        """
        Clean up CUDA resources.

        Args:
            model: The model to clean up.
            tokenizer: The tokenizer to clean up.
            group: The distributed group to clean up.
        """
        import gc

        import torch

        logger.info("Cleaning up CUDA resources")

        # Clean up distributed group
        if group is not None and isinstance(group, (NcclDistributedGroup, GlooDistributedGroup)):
            group.cleanup()

        # Delete model and tokenizer
        if model is not None:
            del model

        if tokenizer is not None:
            del tokenizer

        # Clear CUDA cache and run garbage collection
        gc.collect()
        clear_cuda_cache()
        torch.cuda.synchronize()

        logger.info("CUDA cleanup complete")


# Module-level engine instance (created lazily)
_cuda_engine: CudaEngine | None = None


def get_cuda_engine() -> CudaEngine:
    """
    Get the module-level CUDA engine instance.

    Returns:
        The shared CudaEngine instance.

    Raises:
        RuntimeError: If CUDA is not available.
    """
    global _cuda_engine
    if _cuda_engine is None:
        _cuda_engine = CudaEngine()
    return _cuda_engine

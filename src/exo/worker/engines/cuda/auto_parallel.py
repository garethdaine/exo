# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
# pyright: reportAny=false, reportUnknownParameterType=false
# pyright: reportGeneralTypeIssues=false, reportUntypedBaseClass=false
"""
Model parallelism for CUDA/PyTorch backend.

This module provides tensor and pipeline parallelism strategies for distributing
large language models across multiple GPUs. It mirrors the functionality in the
MLX backend's auto_parallel.py.

Parallelism strategies:
- Tensor Parallelism: Distributes model weights within layers across GPUs.
  Used for models that fit in combined GPU memory but not on a single GPU.
- Pipeline Parallelism: Distributes model layers across GPUs. Each GPU handles
  a subset of layers and passes activations to the next GPU.

Supported model architectures:
- LlamaModel / Ministral3Model
- DeepseekV3Model / DeepseekV32Model
- MiniMaxModel
- Qwen3MoeModel / Glm4MoeModel / Qwen3NextModel
- GptOssModel

Usage:
    from exo.worker.engines.cuda.auto_parallel import (
        tensor_auto_parallel,
        pipeline_auto_parallel,
    )

    # Apply tensor parallelism
    model = tensor_auto_parallel(model, group)

    # Apply pipeline parallelism
    model = pipeline_auto_parallel(model, group, shard_meta)

Note:
    Type checking is relaxed in this module because PyTorch and HuggingFace
    Transformers have complex type signatures that don't work well with
    strict type checking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

from loguru import logger

from exo.shared.types.worker.shards import PipelineShardMetadata

if TYPE_CHECKING:
    import torch
    import torch.distributed as dist
    import torch.nn as nn


# =============================================================================
# Type Protocols
# =============================================================================


class LayerCallable(Protocol):
    """Protocol for callable layers that accept tensors."""

    def __call__(
        self, x: "torch.Tensor", *args: object, **kwargs: object
    ) -> "torch.Tensor": ...


class DistributedGroupProtocol(Protocol):
    """Protocol for distributed communication groups."""

    def rank(self) -> int: ...
    def size(self) -> int: ...


# =============================================================================
# Custom Layer Wrappers
# =============================================================================


class CustomCudaLayer("nn.Module"):
    """Base class for replacing a PyTorch layer with a custom implementation."""

    def __init__(self, original_layer: "nn.Module") -> None:
        import torch.nn as nn

        super().__init__()
        self._original_layer = original_layer

    def __getattr__(self, name: str) -> Any:
        # First try to get from self
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Fall back to original layer
            return getattr(self._original_layer, name)


class PipelineFirstLayer(CustomCudaLayer):
    """
    Wrapper for the first layer in a pipeline stage.

    Receives activations from the previous stage (if not rank 0).
    """

    def __init__(
        self,
        original_layer: "nn.Module",
        rank: int,
        process_group: Any,
    ) -> None:
        super().__init__(original_layer)
        self._rank = rank
        self._process_group = process_group

    def forward(
        self, x: "torch.Tensor", *args: object, **kwargs: object
    ) -> "torch.Tensor":
        import torch
        import torch.distributed as dist

        if self._rank != 0:
            # Receive activations from previous stage
            src_rank = self._rank - 1
            dist.recv(x, src=src_rank, group=self._process_group)

        return self._original_layer(x, *args, **kwargs)


class PipelineLastLayer(CustomCudaLayer):
    """
    Wrapper for the last layer in a pipeline stage.

    Sends activations to the next stage (if not the last rank) and
    gathers results from all stages.
    """

    def __init__(
        self,
        original_layer: "nn.Module",
        rank: int,
        world_size: int,
        process_group: Any,
    ) -> None:
        super().__init__(original_layer)
        self._rank = rank
        self._world_size = world_size
        self._process_group = process_group

    def forward(
        self, x: "torch.Tensor", *args: object, **kwargs: object
    ) -> "torch.Tensor":
        import torch
        import torch.distributed as dist

        output: torch.Tensor = self._original_layer(x, *args, **kwargs)

        if self._rank != self._world_size - 1:
            # Send activations to next stage
            dst_rank = (self._rank + 1) % self._world_size
            dist.send(output, dst=dst_rank, group=self._process_group)

        # All-gather outputs from all stages
        gathered = [torch.zeros_like(output) for _ in range(self._world_size)]
        dist.all_gather(gathered, output, group=self._process_group)

        # Return the last chunk (from the final stage)
        return gathered[-1]


# =============================================================================
# Sharded MoE Layers
# =============================================================================


class ShardedMoE(CustomCudaLayer):
    """
    Wrapper for Mixture-of-Experts layers with tensor parallelism.

    Handles the all-reduce communication pattern for MoE layers where
    experts are sharded across GPUs.
    """

    def __init__(self, original_layer: "nn.Module", process_group: Any) -> None:
        super().__init__(original_layer)
        self._process_group = process_group

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch.distributed as dist

        # Forward through the sharded MoE
        output = self._original_layer(x)

        # All-reduce to combine expert outputs
        dist.all_reduce(output, group=self._process_group)

        return output


# =============================================================================
# Helper Functions
# =============================================================================


def _get_inner_model(model: "nn.Module") -> "nn.Module":
    """Get the inner model from a wrapped model."""
    inner = getattr(model, "model", None)
    if inner is not None:
        return inner

    inner = getattr(model, "transformer", None)
    if inner is not None:
        return inner

    # Return the model itself if no wrapper
    return model


def _get_layers(inner_model: "nn.Module") -> list[Any]:
    """Get the layer list from a model."""
    if hasattr(inner_model, "layers"):
        return inner_model.layers
    elif hasattr(inner_model, "h"):
        return inner_model.h
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


def _set_layers(model: "nn.Module", layers: list[Any]) -> None:
    """Set the layer list on a model."""
    inner_model = _get_inner_model(model)

    if hasattr(inner_model, "layers"):
        inner_model.layers = layers
        # Update model-specific layer count parameters
        if hasattr(inner_model, "num_layers"):
            inner_model.num_layers = len(layers)
        if hasattr(inner_model, "num_hidden_layers"):
            inner_model.num_hidden_layers = len(layers)
    elif hasattr(inner_model, "h"):
        inner_model.h = layers
    else:
        raise ValueError("Model must have either a 'layers' or 'h' attribute")


# =============================================================================
# Sharding Utilities
# =============================================================================


def shard_linear_all_to_sharded(
    linear: "nn.Module",
    process_group: Any,
) -> "nn.Module":
    """
    Shard a linear layer for all-to-sharded communication pattern.

    The input is replicated across all ranks, and each rank computes
    a shard of the output. Used for Q, K, V projections.
    """
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    # Get weight and bias
    weight = linear.weight.data
    bias = linear.bias.data if linear.bias is not None else None

    # Shard along the output dimension
    out_features = weight.shape[0]
    shard_size = out_features // world_size
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size

    # Create new sharded linear layer
    sharded_weight = weight[start_idx:end_idx, :].contiguous()
    sharded_bias = bias[start_idx:end_idx].contiguous() if bias is not None else None

    new_linear = nn.Linear(
        in_features=linear.in_features,
        out_features=shard_size,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    new_linear.weight.data = sharded_weight
    if sharded_bias is not None:
        new_linear.bias.data = sharded_bias

    return new_linear


def shard_linear_sharded_to_all(
    linear: "nn.Module",
    process_group: Any,
) -> "nn.Module":
    """
    Shard a linear layer for sharded-to-all communication pattern.

    Each rank has a shard of the input, and the output is all-reduced
    to produce the full output. Used for output projections.
    """
    import torch
    import torch.distributed as dist
    import torch.nn as nn

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)

    # Get weight and bias
    weight = linear.weight.data
    bias = linear.bias.data if linear.bias is not None else None

    # Shard along the input dimension
    in_features = weight.shape[1]
    shard_size = in_features // world_size
    start_idx = rank * shard_size
    end_idx = start_idx + shard_size

    # Create new sharded linear layer
    sharded_weight = weight[:, start_idx:end_idx].contiguous()

    # Bias is divided by world_size since it will be summed in all-reduce
    sharded_bias = bias / world_size if bias is not None else None

    new_linear = nn.Linear(
        in_features=shard_size,
        out_features=linear.out_features,
        bias=bias is not None,
        device=weight.device,
        dtype=weight.dtype,
    )
    new_linear.weight.data = sharded_weight
    if sharded_bias is not None:
        new_linear.bias.data = sharded_bias

    return new_linear


class AllReduceLinear("nn.Module"):
    """Linear layer wrapper that performs all-reduce after the linear operation."""

    def __init__(self, linear: "nn.Module", process_group: Any) -> None:
        import torch.nn as nn

        super().__init__()
        self.linear = linear
        self._process_group = process_group

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        import torch.distributed as dist

        output = self.linear(x)
        dist.all_reduce(output, group=self._process_group)
        return output


# =============================================================================
# Pipeline Parallelism
# =============================================================================


def pipeline_auto_parallel(
    model: "nn.Module",
    process_group: Any,
    shard_meta: PipelineShardMetadata,
) -> "nn.Module":
    """
    Apply pipeline parallelism to a model.

    Distributes model layers across GPUs. Each GPU handles a subset of layers
    and passes activations to the next GPU.

    Args:
        model: The model to parallelize.
        process_group: The distributed process group.
        shard_meta: Metadata describing the shard assignment.

    Returns:
        The parallelized model with wrapped first/last layers.
    """
    inner_model = _get_inner_model(model)
    layers = _get_layers(inner_model)

    start_layer = shard_meta.start_layer
    end_layer = shard_meta.end_layer
    rank = shard_meta.device_rank
    world_size = shard_meta.world_size

    # Slice layers for this shard
    layers = layers[start_layer:end_layer]

    # Wrap first and last layers
    layers[0] = PipelineFirstLayer(layers[0], rank, process_group)
    layers[-1] = PipelineLastLayer(layers[-1], rank, world_size, process_group)

    _set_layers(model, layers)

    logger.info(
        f"Applied pipeline parallelism: rank {rank}/{world_size}, "
        f"layers [{start_layer}, {end_layer})"
    )

    return model


# =============================================================================
# Tensor Parallelism Strategies
# =============================================================================


class TensorParallelShardingStrategy(ABC):
    """Base class for tensor parallelism sharding strategies."""

    def __init__(self, process_group: Any) -> None:
        import torch.distributed as dist

        self._process_group = process_group
        self._rank = dist.get_rank(process_group)
        self._world_size = dist.get_world_size(process_group)

    @abstractmethod
    def shard_model(self, model: "nn.Module") -> "nn.Module":
        """Apply tensor parallelism sharding to the model."""
        ...

    def _shard_qkv_proj(self, attn: Any) -> None:
        """Shard Q, K, V projections (all-to-sharded pattern)."""
        attn.q_proj = shard_linear_all_to_sharded(attn.q_proj, self._process_group)
        attn.k_proj = shard_linear_all_to_sharded(attn.k_proj, self._process_group)
        attn.v_proj = shard_linear_all_to_sharded(attn.v_proj, self._process_group)

    def _shard_o_proj(self, attn: Any) -> None:
        """Shard output projection (sharded-to-all pattern with all-reduce)."""
        sharded = shard_linear_sharded_to_all(attn.o_proj, self._process_group)
        attn.o_proj = AllReduceLinear(sharded, self._process_group)

    def _shard_mlp(self, mlp: Any) -> None:
        """Shard MLP layers (gate, up are all-to-sharded; down is sharded-to-all)."""
        mlp.gate_proj = shard_linear_all_to_sharded(mlp.gate_proj, self._process_group)
        mlp.up_proj = shard_linear_all_to_sharded(mlp.up_proj, self._process_group)
        sharded_down = shard_linear_sharded_to_all(mlp.down_proj, self._process_group)
        mlp.down_proj = AllReduceLinear(sharded_down, self._process_group)

    def _update_attention_heads(self, attn: Any) -> None:
        """Update attention head counts after sharding."""
        if hasattr(attn, "n_heads"):
            attn.n_heads = attn.n_heads // self._world_size
        if hasattr(attn, "num_heads"):
            attn.num_heads = attn.num_heads // self._world_size
        if hasattr(attn, "num_attention_heads"):
            attn.num_attention_heads = attn.num_attention_heads // self._world_size
        if hasattr(attn, "n_kv_heads") and attn.n_kv_heads is not None:
            attn.n_kv_heads = attn.n_kv_heads // self._world_size
        if hasattr(attn, "num_key_value_heads") and attn.num_key_value_heads is not None:
            attn.num_key_value_heads = attn.num_key_value_heads // self._world_size


class LlamaShardingStrategy(TensorParallelShardingStrategy):
    """Tensor parallelism strategy for Llama-style models."""

    def shard_model(self, model: "nn.Module") -> "nn.Module":
        inner_model = _get_inner_model(model)
        layers = _get_layers(inner_model)

        for layer in layers:
            # Shard attention
            self._shard_qkv_proj(layer.self_attn)
            self._shard_o_proj(layer.self_attn)
            self._update_attention_heads(layer.self_attn)

            # Shard MLP
            self._shard_mlp(layer.mlp)

        logger.info(f"Applied Llama tensor parallelism with world_size={self._world_size}")
        return model


class DeepSeekShardingStrategy(TensorParallelShardingStrategy):
    """Tensor parallelism strategy for DeepSeek models."""

    def shard_model(self, model: "nn.Module") -> "nn.Module":
        inner_model = _get_inner_model(model)
        layers = _get_layers(inner_model)

        for layer in layers:
            attn = layer.self_attn

            # DeepSeek uses LoRA for Q projection in some layers
            if hasattr(attn, "q_lora_rank") and attn.q_lora_rank is not None:
                attn.q_b_proj = shard_linear_all_to_sharded(
                    attn.q_b_proj, self._process_group
                )
            else:
                attn.q_proj = shard_linear_all_to_sharded(
                    attn.q_proj, self._process_group
                )

            # KV projection
            attn.kv_b_proj = shard_linear_all_to_sharded(
                attn.kv_b_proj, self._process_group
            )
            self._shard_o_proj(attn)
            self._update_attention_heads(attn)

            # Shard MLP or MoE
            mlp = layer.mlp
            if hasattr(mlp, "shared_experts"):
                # MoE layer with shared experts
                self._shard_mlp(mlp.shared_experts)
                self._shard_mlp(mlp.switch_mlp)
                layer.mlp = ShardedMoE(mlp, self._process_group)
            else:
                # Standard MLP
                self._shard_mlp(mlp)

        logger.info(
            f"Applied DeepSeek tensor parallelism with world_size={self._world_size}"
        )
        return model


class MiniMaxShardingStrategy(TensorParallelShardingStrategy):
    """Tensor parallelism strategy for MiniMax models."""

    def shard_model(self, model: "nn.Module") -> "nn.Module":
        inner_model = _get_inner_model(model)
        layers = _get_layers(inner_model)

        for layer in layers:
            # Shard attention
            self._shard_qkv_proj(layer.self_attn)
            self._shard_o_proj(layer.self_attn)
            self._update_attention_heads(layer.self_attn)

            # Shard MoE
            moe = layer.block_sparse_moe
            self._shard_mlp(moe.switch_mlp)
            layer.block_sparse_moe = ShardedMoE(moe, self._process_group)

        logger.info(
            f"Applied MiniMax tensor parallelism with world_size={self._world_size}"
        )
        return model


class QwenShardingStrategy(TensorParallelShardingStrategy):
    """Tensor parallelism strategy for Qwen models (including MoE variants)."""

    def shard_model(self, model: "nn.Module") -> "nn.Module":
        inner_model = _get_inner_model(model)
        layers = _get_layers(inner_model)

        for layer in layers:
            # Shard attention
            self._shard_qkv_proj(layer.self_attn)
            self._shard_o_proj(layer.self_attn)
            self._update_attention_heads(layer.self_attn)

            # Shard MLP or MoE
            mlp = layer.mlp
            if hasattr(mlp, "switch_mlp"):
                # MoE layer
                self._shard_mlp(mlp.switch_mlp)
                layer.mlp = ShardedMoE(mlp, self._process_group)
            else:
                # Standard MLP
                self._shard_mlp(mlp)

        logger.info(f"Applied Qwen tensor parallelism with world_size={self._world_size}")
        return model


class GptOssShardingStrategy(TensorParallelShardingStrategy):
    """Tensor parallelism strategy for GPT-OSS models."""

    def shard_model(self, model: "nn.Module") -> "nn.Module":
        inner_model = _get_inner_model(model)
        layers = _get_layers(inner_model)

        for layer in layers:
            attn = layer.self_attn

            # Shard attention
            self._shard_qkv_proj(attn)
            self._shard_o_proj(attn)
            self._update_attention_heads(attn)

            # Update key-value groups
            if hasattr(attn, "num_key_value_groups"):
                attn.num_key_value_groups = (
                    attn.num_attention_heads // attn.num_key_value_heads
                )

            # Shard attention sinks if present
            if hasattr(attn, "sinks"):
                start = attn.num_attention_heads * self._rank
                end = attn.num_attention_heads * (self._rank + 1)
                attn.sinks = attn.sinks[start:end]

            # Shard MoE experts
            moe = layer.mlp
            self._shard_mlp(moe.experts)
            layer.mlp = ShardedMoE(moe, self._process_group)

        logger.info(
            f"Applied GptOss tensor parallelism with world_size={self._world_size}"
        )
        return model


# =============================================================================
# Main Entry Points
# =============================================================================


def tensor_auto_parallel(
    model: "nn.Module",
    process_group: Any,
) -> "nn.Module":
    """
    Apply tensor parallelism to a model based on its architecture.

    This function automatically detects the model architecture and applies
    the appropriate sharding strategy.

    Args:
        model: The model to parallelize.
        process_group: The distributed process group.

    Returns:
        The parallelized model with sharded weights.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    # Check model class name to determine strategy
    model_class = model.__class__.__name__

    # Map model names to strategies
    strategy: TensorParallelShardingStrategy

    if "Llama" in model_class or "Ministral" in model_class:
        strategy = LlamaShardingStrategy(process_group)
    elif "Deepseek" in model_class or "DeepSeek" in model_class:
        strategy = DeepSeekShardingStrategy(process_group)
    elif "MiniMax" in model_class:
        strategy = MiniMaxShardingStrategy(process_group)
    elif "Qwen" in model_class or "Glm4" in model_class:
        strategy = QwenShardingStrategy(process_group)
    elif "GptOss" in model_class:
        strategy = GptOssShardingStrategy(process_group)
    else:
        raise ValueError(
            f"Unsupported model architecture for tensor parallelism: {model_class}. "
            "Supported: Llama, Ministral, DeepSeek, MiniMax, Qwen, Glm4, GptOss"
        )

    return strategy.shard_model(model)


def get_sharding_strategy(
    model: "nn.Module",
    process_group: Any,
) -> TensorParallelShardingStrategy:
    """
    Get the appropriate sharding strategy for a model.

    Args:
        model: The model to get the strategy for.
        process_group: The distributed process group.

    Returns:
        The sharding strategy for the model.

    Raises:
        ValueError: If the model architecture is not supported.
    """
    model_class = model.__class__.__name__

    if "Llama" in model_class or "Ministral" in model_class:
        return LlamaShardingStrategy(process_group)
    elif "Deepseek" in model_class or "DeepSeek" in model_class:
        return DeepSeekShardingStrategy(process_group)
    elif "MiniMax" in model_class:
        return MiniMaxShardingStrategy(process_group)
    elif "Qwen" in model_class or "Glm4" in model_class:
        return QwenShardingStrategy(process_group)
    elif "GptOss" in model_class:
        return GptOssShardingStrategy(process_group)
    else:
        raise ValueError(
            f"Unsupported model architecture: {model_class}. "
            "Supported: Llama, Ministral, DeepSeek, MiniMax, Qwen, Glm4, GptOss"
        )

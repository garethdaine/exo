# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false, reportUnusedImport=false
"""
KV cache management for CUDA/PyTorch backend.

This module provides utilities for managing the key-value cache during
autoregressive generation. The KV cache stores attention keys and values
from previous tokens to avoid redundant computation.

Features:
    - Static cache allocation for predictable memory usage
    - Cache rotation for sliding window attention
    - Memory-efficient cache management

Note:
    Most HuggingFace models handle KV caching internally through the
    `past_key_values` mechanism. This module provides utilities for
    custom cache management when needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    import torch


@dataclass
class CudaKVCacheConfig:
    """Configuration for CUDA KV cache."""

    num_layers: int
    num_heads: int
    head_dim: int
    max_sequence_length: int
    dtype: "torch.dtype"
    device: "torch.device"


class CudaKVCache:
    """
    Key-value cache for CUDA inference.

    This class manages pre-allocated KV cache tensors for efficient
    autoregressive generation. It avoids dynamic memory allocation
    during generation by pre-allocating cache memory.

    Attributes:
        config: The cache configuration.
        key_cache: List of key cache tensors (one per layer).
        value_cache: List of value cache tensors (one per layer).
        current_length: Current number of cached tokens.
    """

    def __init__(self, config: CudaKVCacheConfig) -> None:
        """
        Initialize the KV cache.

        Args:
            config: The cache configuration.
        """
        import torch

        self.config = config
        self.current_length = 0

        # Pre-allocate cache tensors
        # Shape: [batch_size=1, num_heads, max_seq_len, head_dim]
        cache_shape = (
            1,
            config.num_heads,
            config.max_sequence_length,
            config.head_dim,
        )

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

        for _ in range(config.num_layers):
            self.key_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=config.device)
            )
            self.value_cache.append(
                torch.zeros(cache_shape, dtype=config.dtype, device=config.device)
            )

        logger.debug(
            f"Allocated KV cache: {config.num_layers} layers, "
            f"max_seq_len={config.max_sequence_length}"
        )

    def update(
        self,
        layer_idx: int,
        key: "torch.Tensor",
        value: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """
        Update the cache for a layer and return the full cache.

        Args:
            layer_idx: The layer index.
            key: New key tensor [batch, num_heads, seq_len, head_dim].
            value: New value tensor [batch, num_heads, seq_len, head_dim].

        Returns:
            Tuple of (key_cache, value_cache) for the layer.
        """
        seq_len = key.shape[2]

        # Copy new keys and values into cache
        self.key_cache[layer_idx][:, :, self.current_length : self.current_length + seq_len, :] = key
        self.value_cache[layer_idx][:, :, self.current_length : self.current_length + seq_len, :] = value

        # Return the valid portion of the cache
        return (
            self.key_cache[layer_idx][:, :, : self.current_length + seq_len, :],
            self.value_cache[layer_idx][:, :, : self.current_length + seq_len, :],
        )

    def increment_length(self, num_tokens: int = 1) -> None:
        """
        Increment the current cache length.

        Args:
            num_tokens: Number of tokens to add.
        """
        self.current_length += num_tokens

    def reset(self) -> None:
        """Reset the cache for a new generation."""
        self.current_length = 0

    def get_memory_usage(self) -> int:
        """
        Get the memory usage of the cache in bytes.

        Returns:
            Total memory usage in bytes.
        """

        total_bytes = 0
        for cache in self.key_cache + self.value_cache:
            total_bytes += cache.element_size() * cache.numel()
        return total_bytes


def estimate_kv_cache_size(
    num_layers: int,
    num_heads: int,
    head_dim: int,
    max_sequence_length: int,
    dtype_bytes: int = 2,  # float16/bfloat16
) -> int:
    """
    Estimate the memory required for KV cache.

    Args:
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads.
        head_dim: Dimension of each attention head.
        max_sequence_length: Maximum sequence length.
        dtype_bytes: Bytes per element (2 for float16, 4 for float32).

    Returns:
        Estimated memory in bytes.
    """
    # Each layer has key and value caches
    # Shape: [batch=1, num_heads, max_seq_len, head_dim]
    elements_per_cache = num_heads * max_sequence_length * head_dim
    caches_per_layer = 2  # key and value

    total_elements = num_layers * caches_per_layer * elements_per_cache
    return total_elements * dtype_bytes

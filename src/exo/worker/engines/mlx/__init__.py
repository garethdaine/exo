"""
MLX inference engine for Apple Silicon.

This package provides the MLX backend for distributed inference on Apple Silicon
devices. It supports both Ring (TCP/Ethernet) and JACCL (RDMA/Thunderbolt 5)
communication backends for multi-device inference.

Key components:
    - MlxEngine: The main engine class implementing InferenceEngine protocol
    - Model: Type stub for MLX model interface
    - TokenizerWrapper: Type stub for tokenizer interface

Usage:
    from exo.worker.engines.mlx import MlxEngine, get_mlx_engine

    engine = get_mlx_engine()
    # or
    engine = MlxEngine()
"""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

# These are wrapper functions to fix the fact that mlx is not strongly typed in the same way that EXO is.
# For example - MLX has no guarantee of the interface that nn.Module will expose. But we need a guarantee that it has a __call__() function


class Model(nn.Module):
    layers: list[nn.Module]

    def __call__(
        self,
        x: mx.array,
        cache: list[KVCache] | None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array: ...


class Detokenizer:
    def reset(self) -> None: ...
    def add_token(self, token: int) -> None: ...
    def finalize(self) -> None: ...

    @property
    def last_segment(self) -> str: ...


class TokenizerWrapper:
    bos_token: str | None
    eos_token_ids: list[int]
    detokenizer: Detokenizer

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def apply_chat_template(
        self,
        messages_dicts: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str: ...


# Engine exports (after type stubs that use MLX types)
from exo.worker.engines.mlx.engine import (  # noqa: E402
    MlxDistributedGroupWrapper,
    MlxEngine,
    get_mlx_engine,
    mlx_engine,
)

__all__ = [
    # Type stubs
    "Model",
    "Detokenizer",
    "TokenizerWrapper",
    # Engine
    "MlxEngine",
    "MlxDistributedGroupWrapper",
    "get_mlx_engine",
    "mlx_engine",
]

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportAttributeAccessIssue=false, reportUnusedImport=false
# pyright: reportArgumentType=false, reportCallIssue=false, reportAny=false
"""
Token generation for CUDA/PyTorch backend.

This module provides streaming token generation using PyTorch models
and HuggingFace Transformers. It handles:

- Prompt encoding and tokenization
- KV cache management
- Autoregressive token generation
- Stop condition detection
- Streaming output

Usage:
    from exo.worker.engines.cuda.generator import cuda_generate

    for response in cuda_generate(model, tokenizer, task):
        print(response.text, end="", flush=True)

Note:
    Type checking is relaxed in this module because PyTorch and HuggingFace
    Transformers have complex type signatures that don't work well with
    strict type checking.
"""

from __future__ import annotations

import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from loguru import logger

from exo.shared.types.api import GenerationStats
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import ChatCompletionTaskParams
from exo.shared.types.worker.runner_response import GenerationResponse

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def cuda_generate(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    task: ChatCompletionTaskParams,
) -> Generator[GenerationResponse, None, None]:
    """
    Generate tokens using a PyTorch/CUDA model.

    This function performs streaming token generation using the provided
    model and tokenizer. It yields GenerationResponse objects for each
    generated token.

    Args:
        model: The PyTorch model for generation.
        tokenizer: The HuggingFace tokenizer.
        task: The chat completion task parameters.

    Yields:
        GenerationResponse objects containing generated tokens and metadata.
    """
    import torch

    # Get generation parameters
    max_tokens = task.max_tokens or 2048
    temperature = task.temperature or 1.0
    top_p = task.top_p or 1.0

    # Format the prompt using the chat template
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in task.messages
        if msg.content is not None
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    # Get EOS token IDs
    eos_token_ids = set()
    if tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, int):
            eos_token_ids.add(tokenizer.eos_token_id)
        else:
            eos_token_ids.update(tokenizer.eos_token_id)

    # Generation loop
    prompt_tokens = input_ids.shape[1]
    generated_tokens = 0
    start_time = time.perf_counter()

    # Initialize past_key_values for KV cache
    past_key_values = None

    logger.debug(f"Starting CUDA generation: prompt_tokens={prompt_tokens}, max_tokens={max_tokens}")

    with torch.inference_mode():
        for _ in range(max_tokens):
            # Forward pass
            if past_key_values is None:
                # First token: use full prompt
                outputs = model(input_ids, use_cache=True)
            else:
                # Subsequent tokens: use only last token with KV cache
                outputs = model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Apply temperature
            if temperature > 0:
                logits = logits / temperature

            # Apply top-p (nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits = logits.masked_fill(indices_to_remove, float("-inf"))

            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token_id = next_token.item()

            # Append to input_ids
            input_ids = torch.cat([input_ids, next_token], dim=1)
            generated_tokens += 1

            # Decode token
            token_text = tokenizer.decode(
                [next_token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # Check for stop condition
            finish_reason = None
            if next_token_id in eos_token_ids:
                finish_reason = "stop"
            elif generated_tokens >= max_tokens:
                finish_reason = "length"

            # Calculate stats for final token
            stats = None
            if finish_reason is not None:
                elapsed_time = time.perf_counter() - start_time
                tokens_per_second = generated_tokens / elapsed_time if elapsed_time > 0 else 0.0
                stats = GenerationStats(
                    prompt_tokens=prompt_tokens,
                    prompt_tps=prompt_tokens / elapsed_time if elapsed_time > 0 else 0.0,
                    generation_tokens=generated_tokens,
                    generation_tps=tokens_per_second,
                    peak_memory_usage=Memory(),  # TODO: track memory usage
                )

            yield GenerationResponse(
                token=next_token_id,
                text=token_text,
                finish_reason=finish_reason,
                stats=stats,
            )

            if finish_reason is not None:
                break

    logger.debug(f"CUDA generation complete: {generated_tokens} tokens generated")


def warmup_cuda_inference(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    warmup_tokens: int = 10,
) -> int:
    """
    Warm up the CUDA inference pipeline.

    This function performs a short generation to prime GPU caches,
    JIT compilation, and memory allocators.

    Args:
        model: The PyTorch model.
        tokenizer: The HuggingFace tokenizer.
        warmup_tokens: Number of tokens to generate for warmup.

    Returns:
        The number of tokens generated during warmup.
    """
    import torch

    logger.info(f"Warming up CUDA inference ({warmup_tokens} tokens)...")

    # Create a simple warmup prompt
    warmup_prompt = "Hello"
    input_ids = tokenizer.encode(warmup_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)

    tokens_generated = 0
    past_key_values = None

    with torch.inference_mode():
        for _ in range(warmup_tokens):
            if past_key_values is None:
                outputs = model(input_ids, use_cache=True)
            else:
                outputs = model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            tokens_generated += 1

    # Synchronize to ensure warmup is complete
    torch.cuda.synchronize()

    logger.info(f"CUDA warmup complete: {tokens_generated} tokens generated")
    return tokens_generated

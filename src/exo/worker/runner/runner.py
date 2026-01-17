"""
Generic runner implementation using the InferenceEngine abstraction.

This module provides the main runner function that handles inference tasks
using any registered InferenceEngine implementation (MLX, CUDA, Vulkan).

The runner dynamically selects the appropriate engine based on the instance
type and delegates all inference operations to the engine abstraction layer.
"""

import time
from collections.abc import Generator
from contextlib import contextmanager
from functools import cache

from loguru import logger

from exo.shared.types.api import ChatCompletionMessageText
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.common import CommandId
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.models import ModelId
from exo.shared.types.tasks import (
    ChatCompletion,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.instances import (
    BoundInstance,
    CudaGlooInstance,
    CudaNcclInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerFailed,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerStatus,
    RunnerWarmingUp,
)
from exo.utils.channels import MpReceiver, MpSender
from exo.worker.engines import (
    DistributedGroup,
    EngineNotAvailableError,
    EngineNotFoundError,
    InferenceEngine,
    get_engine,
    get_engine_name_for_instance,
)


@contextmanager
def send_error_chunk_on_exception(
    event_sender: MpSender[Event],
    command_id: CommandId,
    model_id: ModelId,
    device_rank: int,
):
    """Context manager to send error chunks when exceptions occur during generation."""
    try:
        yield
    except Exception as e:
        logger.error(e)
        if device_rank == 0:
            event_sender.send(
                ChunkGenerated(
                    command_id=command_id,
                    chunk=TokenChunk(
                        idx=0,
                        model=model_id,
                        text="",
                        token_id=0,
                        finish_reason="error",
                        error_message=str(e),
                    ),
                )
            )


def _get_engine_for_instance(bound_instance: BoundInstance) -> InferenceEngine:
    """
    Get the appropriate inference engine for the given instance type.

    Args:
        bound_instance: The bound instance containing the instance type.

    Returns:
        An InferenceEngine implementation for the instance type.

    Raises:
        RuntimeError: If no suitable engine is available.
    """
    instance = bound_instance.instance
    engine_name = get_engine_name_for_instance(instance)

    try:
        return get_engine(engine_name)
    except (EngineNotFoundError, EngineNotAvailableError) as e:
        raise RuntimeError(
            f"No suitable engine available for instance type "
            f"{type(instance).__name__}: {e}"
        ) from e


def _requires_distributed_init(bound_instance: BoundInstance) -> bool:
    """
    Determine if the instance requires distributed initialization.

    Single-node instances may not need distributed setup.

    Args:
        bound_instance: The bound instance to check.

    Returns:
        True if distributed initialization is required.
    """
    instance = bound_instance.instance

    # MLX instances always require distributed init for now
    if isinstance(instance, (MlxRingInstance, MlxJacclInstance)):
        return True

    # CUDA instances may be single-GPU
    if isinstance(instance, (CudaNcclInstance, CudaGlooInstance)):
        # Multi-GPU or multi-node requires distributed
        shard_count = len(instance.shard_assignments.runner_to_shard)
        return shard_count > 1

    return True


def main(
    bound_instance: BoundInstance,
    event_sender: MpSender[Event],
    task_receiver: MpReceiver[Task],
):
    """
    Main runner function using the InferenceEngine abstraction.

    This function handles the runner lifecycle:
    1. Engine selection based on instance type
    2. Distributed initialization (if needed)
    3. Model loading and sharding
    4. Warmup
    5. Chat completion generation
    6. Cleanup

    Args:
        bound_instance: The bound instance with shard assignments.
        event_sender: Channel to send events back to the worker.
        task_receiver: Channel to receive tasks from the worker.
    """
    instance, runner_id, shard_metadata = (
        bound_instance.instance,
        bound_instance.bound_runner_id,
        bound_instance.bound_shard,
    )

    logger.info("hello from the runner")

    # Handle test-specific immediate exception
    if getattr(shard_metadata, "immediate_exception", False):
        raise Exception("Fake exception - runner failed to spin up.")
    if timeout := getattr(shard_metadata, "should_timeout", 0):
        time.sleep(timeout)

    setup_start_time = time.time()

    # Get the appropriate engine for this instance type
    try:
        engine = _get_engine_for_instance(bound_instance)
        logger.info(f"Selected engine: {type(engine).__name__}")
    except RuntimeError as e:
        logger.error(f"Failed to get engine: {e}")
        event_sender.send(
            RunnerStatusUpdated(
                runner_id=runner_id,
                runner_status=RunnerFailed(error_message=str(e)),
            )
        )
        return

    model = None
    tokenizer = None
    group: DistributedGroup | None = None

    current_status: RunnerStatus = RunnerIdle()
    logger.info("runner created")
    event_sender.send(
        RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
    )

    with task_receiver as tasks:
        for task in tasks:
            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Running)
            )
            event_sender.send(TaskAcknowledged(task_id=task.task_id))

            match task:
                case ConnectToGroup() if isinstance(
                    current_status, (RunnerIdle, RunnerFailed)
                ):
                    logger.info("runner connecting")
                    current_status = RunnerConnecting()
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    # Use engine abstraction for distributed initialization
                    if _requires_distributed_init(bound_instance):
                        group = engine.initialize_distributed(bound_instance)
                        logger.info(
                            f"runner connected (rank {group.rank()}/{group.size()})"
                        )
                    else:
                        logger.info("runner connected (single device, no distributed)")

                    current_status = RunnerConnected()

                case LoadModel() if (
                    isinstance(current_status, RunnerConnected) and group is not None
                ) or (isinstance(current_status, RunnerIdle) and group is None):
                    current_status = RunnerLoading()
                    logger.info("runner loading")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    def on_model_load_timeout() -> None:
                        event_sender.send(
                            RunnerStatusUpdated(
                                runner_id=runner_id,
                                runner_status=RunnerFailed(
                                    error_message="Model loading timed out"
                                ),
                            )
                        )
                        time.sleep(0.5)

                    # Use engine abstraction for model loading
                    model, tokenizer = engine.load_model(
                        bound_instance, group, on_timeout=on_model_load_timeout
                    )

                    current_status = RunnerLoaded()
                    logger.info("runner loaded")

                case StartWarmup() if isinstance(current_status, RunnerLoaded):
                    assert model is not None
                    assert tokenizer is not None
                    current_status = RunnerWarmingUp()
                    logger.info("runner warming up")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    logger.info(f"warming up inference for instance: {instance}")

                    # Use engine abstraction for warmup
                    toks = engine.warmup(model, tokenizer)

                    logger.info(f"warmed up by generating {toks} tokens")
                    logger.info(
                        f"runner initialized in {time.time() - setup_start_time} seconds"
                    )
                    current_status = RunnerReady()
                    logger.info("runner ready")

                case ChatCompletion(task_params=task_params, command_id=command_id) if (
                    isinstance(current_status, RunnerReady)
                ):
                    logger.info(f"received chat request: {str(task)[:500]}")
                    current_status = RunnerRunning()
                    logger.info("runner running")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )
                    with send_error_chunk_on_exception(
                        event_sender,
                        command_id,
                        shard_metadata.model_meta.model_id,
                        shard_metadata.device_rank,
                    ):
                        assert model is not None
                        assert tokenizer is not None
                        assert task_params.messages[0].content is not None
                        _check_for_debug_prompts(task_params.messages[0].content)

                        # Use engine abstraction for generation
                        generator = engine.generate(model, tokenizer, task_params)

                        # Apply post-processing for specific model types
                        generator = _apply_model_specific_parsing(
                            generator, model, engine
                        )

                        for response in generator:
                            match response:
                                case GenerationResponse():
                                    if shard_metadata.device_rank == 0:
                                        event_sender.send(
                                            ChunkGenerated(
                                                command_id=command_id,
                                                chunk=TokenChunk(
                                                    idx=response.token,
                                                    model=shard_metadata.model_meta.model_id,
                                                    text=response.text,
                                                    token_id=response.token,
                                                    finish_reason=response.finish_reason,
                                                    stats=response.stats,
                                                ),
                                            )
                                        )

                    current_status = RunnerReady()
                    logger.info("runner ready")

                case Shutdown():
                    current_status = RunnerShuttingDown()
                    logger.info("runner shutting down")
                    event_sender.send(
                        RunnerStatusUpdated(
                            runner_id=runner_id, runner_status=current_status
                        )
                    )

                    # Use engine abstraction for cleanup
                    engine.cleanup(model, tokenizer, group)

                    current_status = RunnerShutdown()

                case _:
                    raise ValueError(
                        f"Received {task.__class__.__name__} outside of state machine in {current_status=}"
                    )

            event_sender.send(
                TaskStatusUpdated(task_id=task.task_id, task_status=TaskStatus.Complete)
            )
            event_sender.send(
                RunnerStatusUpdated(runner_id=runner_id, runner_status=current_status)
            )
            if isinstance(current_status, RunnerShutdown):
                break


def _apply_model_specific_parsing(
    generator: Generator[GenerationResponse, None, None],
    model: object,
    engine: InferenceEngine,
) -> Generator[GenerationResponse, None, None]:
    """
    Apply model-specific parsing to the generation output.

    Some models (like GPT-OSS) require special output parsing.
    This function checks if the model needs special handling and
    wraps the generator accordingly.

    Args:
        generator: The base generation response generator.
        model: The loaded model.
        engine: The inference engine being used.

    Yields:
        Possibly transformed GenerationResponse objects.
    """
    # Check if this is an MLX engine with GptOssModel
    try:
        from mlx_lm.models.gpt_oss import Model as GptOssModel

        if isinstance(model, GptOssModel):
            return parse_gpt_oss(generator)
    except ImportError:
        pass

    return generator


@cache
def get_gpt_oss_encoding():
    """Get the GPT-OSS encoding (cached)."""
    from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
        HarmonyEncodingName,
        load_harmony_encoding,
    )

    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return encoding


def parse_gpt_oss(
    responses: Generator[GenerationResponse, None, None],
) -> Generator[GenerationResponse, None, None]:
    """
    Parse GPT-OSS model output to match expected format.

    GPT-OSS uses a special encoding with thinking/analysis channels
    that need to be converted to <think> tags.

    Args:
        responses: The raw generation responses.

    Yields:
        Transformed GenerationResponse objects with think tags.
    """
    from openai_harmony import (  # pyright: ignore[reportMissingTypeStubs]
        Role,
        StreamableParser,
    )

    encoding = get_gpt_oss_encoding()
    stream = StreamableParser(encoding, role=Role.ASSISTANT)
    thinking = False

    for response in responses:
        stream.process(response.token)

        delta = stream.last_content_delta
        ch = stream.current_channel

        if ch == "analysis" and not thinking:
            thinking = True
            yield response.model_copy(update={"text": "<think>"})

        if ch != "analysis" and thinking:
            thinking = False
            yield response.model_copy(update={"text": "</think>"})

        if delta:
            yield response.model_copy(update={"text": delta})

        if response.finish_reason is not None:
            if thinking:
                yield response.model_copy(update={"text": "</think>"})
            yield response
            break


# Debug prompt constants
EXO_RUNNER_MUST_FAIL = "EXO RUNNER MUST FAIL"
EXO_RUNNER_MUST_OOM = "EXO RUNNER MUST OOM"
EXO_RUNNER_MUST_TIMEOUT = "EXO RUNNER MUST TIMEOUT"


def _check_for_debug_prompts(
    prompt: str | ChatCompletionMessageText | list[ChatCompletionMessageText],
):
    """
    Check for debug prompts that trigger special behavior.

    This function is used for testing purposes to simulate failures,
    OOM conditions, and timeouts.

    Args:
        prompt: The user prompt to check.
    """
    if isinstance(prompt, list):
        if len(prompt) == 0:
            logger.debug("Empty message prompt received in debug prompt")
            return
        prompt = prompt[0]

    if isinstance(prompt, ChatCompletionMessageText):
        prompt = prompt.text

    if EXO_RUNNER_MUST_FAIL in prompt:
        logger.info("raising exception")
        raise Exception("Artificial runner exception - for testing purposes only.")
    if EXO_RUNNER_MUST_OOM in prompt:
        # Import the appropriate OOM function based on available engine
        try:
            from exo.worker.engines.mlx.utils_mlx import mlx_force_oom

            mlx_force_oom()
        except ImportError:
            logger.warning("MLX not available for OOM simulation")
    if EXO_RUNNER_MUST_TIMEOUT in prompt:
        time.sleep(100)

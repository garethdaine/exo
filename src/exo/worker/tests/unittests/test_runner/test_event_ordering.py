# Check tasks are complete before runner is ever ready.
from collections.abc import Iterable
from typing import Callable
from unittest.mock import MagicMock

import pytest

import exo.worker.runner.runner as runner_module
from exo.shared.types.api import ChatCompletionMessage
from exo.shared.types.chunks import TokenChunk
from exo.shared.types.events import (
    ChunkGenerated,
    Event,
    RunnerStatusUpdated,
    TaskAcknowledged,
    TaskStatusUpdated,
)
from exo.shared.types.tasks import (
    ChatCompletion,
    ChatCompletionTaskParams,
    ConnectToGroup,
    LoadModel,
    Shutdown,
    StartWarmup,
    Task,
    TaskStatus,
)
from exo.shared.types.worker.runner_response import GenerationResponse
from exo.shared.types.worker.runners import (
    RunnerConnected,
    RunnerConnecting,
    RunnerIdle,
    RunnerLoaded,
    RunnerLoading,
    RunnerReady,
    RunnerRunning,
    RunnerShutdown,
    RunnerShuttingDown,
    RunnerWarmingUp,
)
from exo.utils.channels import mp_channel

from ...constants import (
    CHAT_COMPLETION_TASK_ID,
    COMMAND_1_ID,
    INITIALIZATION_TASK_ID,
    INSTANCE_1_ID,
    LOAD_TASK_ID,
    MODEL_A_ID,
    NODE_A,
    RUNNER_1_ID,
    SHUTDOWN_TASK_ID,
    WARMUP_TASK_ID,
)
from ..conftest import get_bound_mlx_ring_instance


def make_nothin[T, U, V](res: T) -> Callable[[], T]:
    def nothin(*_1: U, **_2: V) -> T:
        return res

    return nothin


nothin = make_nothin(None)


INIT_TASK = ConnectToGroup(
    task_id=INITIALIZATION_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

LOAD_TASK = LoadModel(
    task_id=LOAD_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

WARMUP_TASK = StartWarmup(
    task_id=WARMUP_TASK_ID,
    instance_id=INSTANCE_1_ID,
)

SHUTDOWN_TASK = Shutdown(
    task_id=SHUTDOWN_TASK_ID,
    instance_id=INSTANCE_1_ID,
    runner_id=RUNNER_1_ID,
)

CHAT_PARAMS = ChatCompletionTaskParams(
    model=str(MODEL_A_ID),
    messages=[ChatCompletionMessage(role="user", content="hello")],
    stream=True,
    max_tokens=4,
    temperature=0.0,
)

CHAT_TASK = ChatCompletion(
    task_id=CHAT_COMPLETION_TASK_ID,
    command_id=COMMAND_1_ID,
    task_params=CHAT_PARAMS,
    instance_id=INSTANCE_1_ID,
)


def assert_events_equal(test_events: Iterable[Event], true_events: Iterable[Event]):
    for test_event, true_event in zip(test_events, true_events, strict=True):
        test_event.event_id = true_event.event_id
        assert test_event == true_event, f"{test_event} != {true_event}"


def _create_mock_engine() -> MagicMock:
    """Create a mock InferenceEngine for testing."""
    mock_engine: MagicMock = MagicMock()

    # Mock distributed group
    mock_group: MagicMock = MagicMock()
    mock_group.rank.return_value = 0  # pyright: ignore[reportAny]
    mock_group.size.return_value = 1  # pyright: ignore[reportAny]
    mock_engine.initialize_distributed.return_value = mock_group  # pyright: ignore[reportAny]

    # Mock load_model - return (model, tokenizer)
    mock_engine.load_model.return_value = (MagicMock(), MagicMock())  # pyright: ignore[reportAny]

    # Mock warmup
    mock_engine.warmup.return_value = 1  # pyright: ignore[reportAny]

    # Mock generate - yield a generation response
    def fake_generate(*_1: object, **_2: object):
        yield GenerationResponse(token=0, text="hi", finish_reason="stop")

    mock_engine.generate.side_effect = fake_generate  # pyright: ignore[reportAny]

    # Mock cleanup
    mock_engine.cleanup.return_value = None  # pyright: ignore[reportAny]

    return mock_engine


@pytest.fixture
def patch_out_engine(monkeypatch: pytest.MonkeyPatch):
    """Patch out the engine selection to use a mock engine."""
    mock_engine = _create_mock_engine()

    # Patch _get_engine_for_instance to return our mock
    monkeypatch.setattr(
        runner_module, "_get_engine_for_instance", make_nothin(mock_engine)
    )

    # Patch _check_for_debug_prompts to do nothing
    monkeypatch.setattr(runner_module, "_check_for_debug_prompts", nothin)


def _run(tasks: Iterable[Task]):
    bound_instance = get_bound_mlx_ring_instance(
        instance_id=INSTANCE_1_ID,
        model_id=MODEL_A_ID,
        runner_id=RUNNER_1_ID,
        node_id=NODE_A,
    )

    task_sender, task_receiver = mp_channel[Task]()
    event_sender, event_receiver = mp_channel[Event]()

    with task_sender, event_receiver:
        for t in tasks:
            task_sender.send(t)

        # worst monkeypatch known to man
        # this is some c++ nonsense
        event_sender.close = nothin
        event_sender.join = nothin
        task_receiver.close = nothin
        task_receiver.join = nothin

        runner_module.main(bound_instance, event_sender, task_receiver)

        return event_receiver.collect()


def test_events_processed_in_correct_order(patch_out_engine: pytest.MonkeyPatch):
    events = _run([INIT_TASK, LOAD_TASK, WARMUP_TASK, CHAT_TASK, SHUTDOWN_TASK])

    expected_chunk = ChunkGenerated(
        command_id=COMMAND_1_ID,
        chunk=TokenChunk(
            idx=0,
            model=MODEL_A_ID,
            text="hi",
            token_id=0,
            finish_reason="stop",
        ),
    )

    assert_events_equal(
        events,
        [
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerIdle()),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Running
            ),
            TaskAcknowledged(task_id=INITIALIZATION_TASK_ID),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerConnecting()
            ),
            TaskStatusUpdated(
                task_id=INITIALIZATION_TASK_ID, task_status=TaskStatus.Complete
            ),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerConnected()),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=LOAD_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoading()),
            TaskStatusUpdated(task_id=LOAD_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerLoaded()),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=WARMUP_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerWarmingUp()),
            TaskStatusUpdated(task_id=WARMUP_TASK_ID, task_status=TaskStatus.Complete),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Running
            ),
            TaskAcknowledged(task_id=CHAT_COMPLETION_TASK_ID),
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerRunning()),
            expected_chunk,
            TaskStatusUpdated(
                task_id=CHAT_COMPLETION_TASK_ID, task_status=TaskStatus.Complete
            ),
            # CHAT COMPLETION TASK SHOULD COMPLETE BEFORE RUNNER READY
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerReady()),
            TaskStatusUpdated(task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Running),
            TaskAcknowledged(task_id=SHUTDOWN_TASK_ID),
            RunnerStatusUpdated(
                runner_id=RUNNER_1_ID, runner_status=RunnerShuttingDown()
            ),
            TaskStatusUpdated(
                task_id=SHUTDOWN_TASK_ID, task_status=TaskStatus.Complete
            ),
            # SPECIAL EXCEPTION FOR RUNNER SHUTDOWN
            RunnerStatusUpdated(runner_id=RUNNER_1_ID, runner_status=RunnerShutdown()),
        ],
    )

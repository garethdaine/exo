"""
Engine registry for inference backend discovery and instantiation.

This module provides a registry pattern for inference engines, allowing
dynamic registration and retrieval of backends based on platform capabilities.

The registry enables:
- Runtime engine discovery based on available hardware
- Dynamic loading of engine implementations
- Platform-specific engine selection (MLX for macOS, CUDA for Linux/Windows)

Usage:
    from exo.worker.engines.registry import get_engine, register_engine

    # Register an engine (typically done at module import)
    @register_engine("mlx")
    class MlxEngine:
        ...

    # Get an engine by name
    engine = get_engine("mlx")
"""

import sys
from collections.abc import Callable
from typing import TypeVar

from exo.worker.engines.base import InferenceEngine

# Type variable for engine classes
EngineType = TypeVar("EngineType", bound=InferenceEngine)

# Global registry of engine factories
_ENGINE_REGISTRY: dict[str, Callable[[], InferenceEngine]] = {}


class EngineNotFoundError(Exception):
    """Raised when a requested engine is not found in the registry."""

    def __init__(self, engine_name: str, available: list[str]) -> None:
        self.engine_name = engine_name
        self.available = available
        super().__init__(
            f"Engine '{engine_name}' not found. Available engines: {available}"
        )


class EngineNotAvailableError(Exception):
    """Raised when an engine is not available on the current platform."""

    def __init__(self, engine_name: str, reason: str) -> None:
        self.engine_name = engine_name
        self.reason = reason
        super().__init__(f"Engine '{engine_name}' not available: {reason}")


def register_engine(
    name: str,
) -> Callable[[type[EngineType]], type[EngineType]]:
    """
    Decorator to register an engine factory in the global registry.

    Args:
        name: The name to register the engine under (e.g., "mlx", "cuda").

    Returns:
        A decorator that registers the engine class.

    Example:
        @register_engine("mlx")
        class MlxEngine:
            def __init__(self) -> None:
                ...
    """

    def decorator(engine_class: type[EngineType]) -> type[EngineType]:
        _ENGINE_REGISTRY[name] = engine_class
        return engine_class

    return decorator


def register_engine_factory(
    name: str,
    factory: Callable[[], InferenceEngine],
) -> None:
    """
    Register an engine factory function in the global registry.

    This is useful for engines that require special initialization or
    conditional loading.

    Args:
        name: The name to register the engine under.
        factory: A callable that returns an InferenceEngine instance.
    """
    _ENGINE_REGISTRY[name] = factory


def get_engine(name: str) -> InferenceEngine:
    """
    Get an engine instance by name.

    Args:
        name: The registered name of the engine (e.g., "mlx", "cuda").

    Returns:
        An InferenceEngine instance.

    Raises:
        EngineNotFoundError: If the engine is not registered.
        EngineNotAvailableError: If the engine cannot be instantiated.
    """
    if name not in _ENGINE_REGISTRY:
        raise EngineNotFoundError(name, list(_ENGINE_REGISTRY.keys()))

    try:
        return _ENGINE_REGISTRY[name]()
    except ImportError as e:
        raise EngineNotAvailableError(name, f"Missing dependency: {e}") from e
    except Exception as e:
        raise EngineNotAvailableError(name, str(e)) from e


def list_available_engines() -> list[str]:
    """
    List all registered engine names.

    Returns:
        List of engine names that are registered.
    """
    return list(_ENGINE_REGISTRY.keys())


def is_engine_available(name: str) -> bool:
    """
    Check if an engine is registered and can be instantiated.

    Args:
        name: The engine name to check.

    Returns:
        True if the engine is available, False otherwise.
    """
    if name not in _ENGINE_REGISTRY:
        return False

    try:
        # Try to instantiate the engine to verify it's available
        _ = _ENGINE_REGISTRY[name]()
        return True
    except Exception:
        return False


def get_default_engine_for_platform() -> str:
    """
    Determine the default engine for the current platform.

    Returns:
        The name of the default engine for this platform:
        - "mlx" for macOS (Darwin)
        - "cuda" for Linux/Windows (if available)
        - Raises if no suitable engine is available

    Raises:
        EngineNotAvailableError: If no suitable engine is available.
    """
    platform = sys.platform

    if platform == "darwin":
        # macOS -> prefer MLX
        if is_engine_available("mlx"):
            return "mlx"
    elif platform in ("linux", "win32"):
        # Linux/Windows -> prefer CUDA
        if is_engine_available("cuda"):
            return "cuda"
        # Fall back to Vulkan if available
        if is_engine_available("vulkan"):
            return "vulkan"

    # List available engines for error message
    available = [name for name in _ENGINE_REGISTRY if is_engine_available(name)]
    if available:
        return available[0]

    raise EngineNotAvailableError(
        "default",
        f"No suitable engine available for platform '{platform}'. "
        f"Registered engines: {list(_ENGINE_REGISTRY.keys())}",
    )


# Auto-register MLX engine on macOS
def _register_default_engines() -> None:
    """Register default engines based on platform availability."""
    if sys.platform == "darwin":
        try:
            from exo.worker.engines.mlx.engine import MlxEngine

            register_engine_factory("mlx", MlxEngine)
        except ImportError:
            pass  # MLX not available

    # Future: Register CUDA engine on Linux/Windows
    # Future: Register Vulkan engine for cross-platform GPU support


# Register engines on module import
_register_default_engines()

# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportAttributeAccessIssue=false
"""
Cross-engine tensor serialization for heterogeneous clusters.

This module provides tensor serialization protocols to enable communication between
different inference backends (MLX, CUDA/PyTorch) over the network. This is essential
for pipeline parallelism across heterogeneous nodes.

The serialization uses NumPy as an intermediate format, which both MLX and PyTorch
can efficiently convert to/from. The wire format includes metadata (shape, dtype)
to enable correct deserialization on the receiving end.

Performance Considerations:
    - 10G Ethernet: ~1.25 GB/s theoretical throughput
    - Llama-3-70B hidden state: ~16KB per token (FP16)
    - Target: < 5ms serialization overhead per transfer

Usage:
    from exo.worker.engines.serialization import TensorSerializer

    # Serialize an MLX tensor
    serializer = TensorSerializer()
    data = serializer.serialize_mlx(mlx_tensor)

    # Deserialize to PyTorch CUDA tensor
    cuda_tensor = serializer.deserialize_to_cuda(data)

Note:
    Type checking is relaxed in this module because it interfaces with optional
    dependencies (PyTorch) that may not be installed on all platforms.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import DTypeLike, NDArray

# TYPE_CHECKING is used to avoid importing torch at runtime on macOS
# where PyTorch may not be installed
_ = TYPE_CHECKING  # Silence unused import warning

# Type alias for supported frameworks
FrameworkType = Literal["mlx", "cuda", "numpy"]


class TensorDtype(str, Enum):
    """Supported tensor data types for cross-engine serialization."""

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT64 = "int64"
    INT32 = "int32"
    INT16 = "int16"
    INT8 = "int8"
    UINT8 = "uint8"

    def to_numpy_dtype(self) -> DTypeLike:
        """Convert to numpy dtype."""
        mapping: dict[TensorDtype, DTypeLike] = {
            TensorDtype.FLOAT32: np.float32,
            TensorDtype.FLOAT16: np.float16,
            TensorDtype.BFLOAT16: np.float16,  # NumPy doesn't support bfloat16 natively
            TensorDtype.INT64: np.int64,
            TensorDtype.INT32: np.int32,
            TensorDtype.INT16: np.int16,
            TensorDtype.INT8: np.int8,
            TensorDtype.UINT8: np.uint8,
        }
        return mapping[self]

    @classmethod
    def from_numpy_dtype(cls, dtype: np.dtype[np.generic]) -> "TensorDtype":
        """Convert from numpy dtype."""
        dtype_str = str(dtype)
        mapping: dict[str, TensorDtype] = {
            "float32": TensorDtype.FLOAT32,
            "float16": TensorDtype.FLOAT16,
            "int64": TensorDtype.INT64,
            "int32": TensorDtype.INT32,
            "int16": TensorDtype.INT16,
            "int8": TensorDtype.INT8,
            "uint8": TensorDtype.UINT8,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported numpy dtype: {dtype_str}")
        return mapping[dtype_str]


@dataclass(frozen=True)
class TensorMetadata:
    """
    Metadata for serialized tensors.

    This metadata is sent alongside the tensor data to enable correct
    deserialization on the receiving end.
    """

    shape: tuple[int, ...]
    dtype: TensorDtype
    source_framework: FrameworkType
    byte_size: int

    def to_bytes(self) -> bytes:
        """
        Serialize metadata to bytes.

        Format:
            - 4 bytes: number of dimensions
            - 4 bytes per dimension: shape
            - 1 byte: dtype enum value
            - 1 byte: source framework enum value
            - 8 bytes: byte size
        """
        # Number of dimensions
        data = len(self.shape).to_bytes(4, "little")

        # Shape dimensions
        for dim in self.shape:
            data += dim.to_bytes(4, "little")

        # Dtype (index in enum)
        dtype_values = list(TensorDtype)
        data += dtype_values.index(self.dtype).to_bytes(1, "little")

        # Source framework
        framework_map: dict[FrameworkType, int] = {"mlx": 0, "cuda": 1, "numpy": 2}
        data += framework_map[self.source_framework].to_bytes(1, "little")

        # Byte size
        data += self.byte_size.to_bytes(8, "little")

        return data

    @classmethod
    def from_bytes(cls, data: bytes) -> tuple["TensorMetadata", int]:
        """
        Deserialize metadata from bytes.

        Args:
            data: The bytes to deserialize from.

        Returns:
            A tuple of (TensorMetadata, bytes_consumed).
        """
        offset = 0

        # Number of dimensions
        num_dims = int.from_bytes(data[offset : offset + 4], "little")
        offset += 4

        # Shape dimensions
        shape: list[int] = []
        for _ in range(num_dims):
            shape.append(int.from_bytes(data[offset : offset + 4], "little"))
            offset += 4

        # Dtype
        dtype_index = int.from_bytes(data[offset : offset + 1], "little")
        dtype = list(TensorDtype)[dtype_index]
        offset += 1

        # Source framework
        framework_index = int.from_bytes(data[offset : offset + 1], "little")
        framework_map: dict[int, FrameworkType] = {0: "mlx", 1: "cuda", 2: "numpy"}
        source_framework = framework_map[framework_index]
        offset += 1

        # Byte size
        byte_size = int.from_bytes(data[offset : offset + 8], "little")
        offset += 8

        return (
            cls(
                shape=tuple(shape),
                dtype=dtype,
                source_framework=source_framework,
                byte_size=byte_size,
            ),
            offset,
        )


@dataclass(frozen=True)
class SerializedTensor:
    """
    A serialized tensor with metadata.

    This is the wire format for tensor transfer between nodes.
    """

    metadata: TensorMetadata
    data: bytes

    def to_bytes(self) -> bytes:
        """Serialize to bytes for network transfer."""
        metadata_bytes = self.metadata.to_bytes()
        # Prefix with metadata length for parsing
        return (
            len(metadata_bytes).to_bytes(4, "little") + metadata_bytes + self.data
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "SerializedTensor":
        """Deserialize from bytes."""
        # Read metadata length
        metadata_len = int.from_bytes(data[:4], "little")

        # Parse metadata
        metadata, _ = TensorMetadata.from_bytes(data[4 : 4 + metadata_len])

        # Extract tensor data
        tensor_data = data[4 + metadata_len : 4 + metadata_len + metadata.byte_size]

        return cls(metadata=metadata, data=tensor_data)


@runtime_checkable
class TensorSerializerProtocol(Protocol):
    """Protocol for tensor serialization implementations."""

    def serialize(self, tensor: object, source_framework: FrameworkType) -> SerializedTensor:
        """Serialize a tensor from any supported framework."""
        ...

    def deserialize(
        self, serialized: SerializedTensor, target_framework: FrameworkType
    ) -> object:
        """Deserialize a tensor to any supported framework."""
        ...


class TensorSerializer:
    """
    Cross-engine tensor serializer using NumPy as intermediate format.

    This class handles serialization and deserialization of tensors between
    MLX, PyTorch/CUDA, and NumPy formats. It uses NumPy as a common intermediate
    representation since both MLX and PyTorch can efficiently convert to/from it.

    Example:
        serializer = TensorSerializer()

        # MLX to CUDA transfer
        mlx_tensor = mx.array([1, 2, 3])
        serialized = serializer.serialize_mlx(mlx_tensor)
        cuda_tensor = serializer.deserialize_to_cuda(serialized)

        # CUDA to MLX transfer
        cuda_tensor = torch.tensor([1, 2, 3]).cuda()
        serialized = serializer.serialize_cuda(cuda_tensor)
        mlx_tensor = serializer.deserialize_to_mlx(serialized)
    """

    def serialize_numpy(self, array: NDArray[np.generic]) -> SerializedTensor:
        """
        Serialize a NumPy array.

        Args:
            array: The NumPy array to serialize.

        Returns:
            SerializedTensor containing the array data and metadata.
        """
        # Ensure contiguous memory layout for efficient serialization
        if not array.flags["C_CONTIGUOUS"]:
            array = np.ascontiguousarray(array)

        data = array.tobytes()
        metadata = TensorMetadata(
            shape=array.shape,
            dtype=TensorDtype.from_numpy_dtype(array.dtype),
            source_framework="numpy",
            byte_size=len(data),
        )
        return SerializedTensor(metadata=metadata, data=data)

    def deserialize_to_numpy(self, serialized: SerializedTensor) -> NDArray[np.generic]:
        """
        Deserialize to a NumPy array.

        Args:
            serialized: The serialized tensor.

        Returns:
            NumPy array with the tensor data.
        """
        dtype = serialized.metadata.dtype.to_numpy_dtype()
        array: NDArray[np.generic] = np.frombuffer(
            serialized.data, dtype=dtype
        ).reshape(serialized.metadata.shape)
        return array

    def serialize_mlx(self, tensor: object) -> SerializedTensor:
        """
        Serialize an MLX array.

        Args:
            tensor: The MLX array (mx.array) to serialize.

        Returns:
            SerializedTensor containing the array data and metadata.

        Note:
            This method imports mlx.core lazily to avoid import errors
            on non-macOS platforms.
        """
        import mlx.core as mx

        # Type narrowing for the MLX array
        if not isinstance(tensor, mx.array):
            raise TypeError(f"Expected mx.array, got {type(tensor)}")

        # Convert to numpy (this is efficient in MLX)
        np_array: NDArray[np.generic] = np.array(tensor)

        # Handle bfloat16 specially
        mlx_dtype_str = str(tensor.dtype)
        is_bfloat16 = "bfloat16" in mlx_dtype_str

        data = np_array.tobytes()
        metadata = TensorMetadata(
            shape=tuple(tensor.shape),
            dtype=TensorDtype.BFLOAT16 if is_bfloat16 else TensorDtype.from_numpy_dtype(np_array.dtype),
            source_framework="mlx",
            byte_size=len(data),
        )
        return SerializedTensor(metadata=metadata, data=data)

    def deserialize_to_mlx(self, serialized: SerializedTensor) -> object:
        """
        Deserialize to an MLX array.

        Args:
            serialized: The serialized tensor.

        Returns:
            MLX array (mx.array) with the tensor data.

        Note:
            This method imports mlx.core lazily to avoid import errors
            on non-macOS platforms.
        """
        import mlx.core as mx

        # First deserialize to numpy
        np_array = self.deserialize_to_numpy(serialized)

        # Convert to MLX array
        mlx_array = mx.array(np_array)

        # Handle bfloat16 conversion if needed
        if serialized.metadata.dtype == TensorDtype.BFLOAT16:
            mlx_array = mlx_array.astype(mx.bfloat16)

        return mlx_array

    def serialize_cuda(self, tensor: object) -> SerializedTensor:
        """
        Serialize a PyTorch CUDA tensor.

        Args:
            tensor: The PyTorch tensor (torch.Tensor) to serialize.

        Returns:
            SerializedTensor containing the tensor data and metadata.

        Note:
            This method imports torch lazily to avoid import errors
            on platforms without PyTorch/CUDA.
        """
        import torch

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")

        # Move to CPU and convert to numpy
        # contiguous() ensures efficient memory layout
        cpu_tensor = tensor.detach().cpu().contiguous()
        np_array: NDArray[np.generic] = cpu_tensor.numpy()

        # Handle bfloat16 specially (PyTorch supports it, numpy doesn't)
        is_bfloat16 = tensor.dtype == torch.bfloat16
        if is_bfloat16:
            # Convert to float16 for serialization
            np_array = cpu_tensor.to(torch.float16).numpy()

        data = np_array.tobytes()
        metadata = TensorMetadata(
            shape=tuple(tensor.shape),
            dtype=TensorDtype.BFLOAT16 if is_bfloat16 else TensorDtype.from_numpy_dtype(np_array.dtype),
            source_framework="cuda",
            byte_size=len(data),
        )
        return SerializedTensor(metadata=metadata, data=data)

    def deserialize_to_cuda(
        self, serialized: SerializedTensor, device: str = "cuda"
    ) -> object:
        """
        Deserialize to a PyTorch CUDA tensor.

        Args:
            serialized: The serialized tensor.
            device: The target CUDA device (default: "cuda").

        Returns:
            PyTorch tensor on the specified CUDA device.

        Note:
            This method imports torch lazily to avoid import errors
            on platforms without PyTorch/CUDA.
        """
        import torch

        # First deserialize to numpy
        np_array = self.deserialize_to_numpy(serialized)

        # Convert to PyTorch tensor
        torch_tensor = torch.from_numpy(np_array.copy())

        # Handle bfloat16 conversion if needed
        if serialized.metadata.dtype == TensorDtype.BFLOAT16:
            torch_tensor = torch_tensor.to(torch.bfloat16)

        # Move to CUDA device
        return torch_tensor.to(device)

    def serialize(
        self, tensor: object, source_framework: FrameworkType
    ) -> SerializedTensor:
        """
        Serialize a tensor from any supported framework.

        Args:
            tensor: The tensor to serialize.
            source_framework: The framework the tensor is from.

        Returns:
            SerializedTensor containing the tensor data and metadata.
        """
        if source_framework == "mlx":
            return self.serialize_mlx(tensor)
        elif source_framework == "cuda":
            return self.serialize_cuda(tensor)
        elif source_framework == "numpy":
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Expected np.ndarray, got {type(tensor)}")
            return self.serialize_numpy(tensor)
        else:
            raise ValueError(f"Unsupported source framework: {source_framework}")

    def deserialize(
        self, serialized: SerializedTensor, target_framework: FrameworkType
    ) -> object:
        """
        Deserialize a tensor to any supported framework.

        Args:
            serialized: The serialized tensor.
            target_framework: The target framework to deserialize to.

        Returns:
            Tensor in the target framework's format.
        """
        if target_framework == "mlx":
            return self.deserialize_to_mlx(serialized)
        elif target_framework == "cuda":
            return self.deserialize_to_cuda(serialized)
        elif target_framework == "numpy":
            return self.deserialize_to_numpy(serialized)
        else:
            raise ValueError(f"Unsupported target framework: {target_framework}")


# Module-level serializer instance for convenience
tensor_serializer = TensorSerializer()

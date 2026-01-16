"""Tests for cross-engine tensor serialization."""

import numpy as np
import pytest

from exo.worker.engines.serialization import (
    SerializedTensor,
    TensorDtype,
    TensorMetadata,
    TensorSerializer,
)


class TestTensorMetadata:
    """Tests for TensorMetadata serialization."""

    def test_metadata_roundtrip(self) -> None:
        """Test that metadata can be serialized and deserialized."""
        metadata = TensorMetadata(
            shape=(2, 3, 4),
            dtype=TensorDtype.FLOAT16,
            source_framework="mlx",
            byte_size=48,
        )
        data = metadata.to_bytes()
        restored, consumed = TensorMetadata.from_bytes(data)

        assert restored.shape == metadata.shape
        assert restored.dtype == metadata.dtype
        assert restored.source_framework == metadata.source_framework
        assert restored.byte_size == metadata.byte_size
        assert consumed == len(data)

    def test_metadata_different_shapes(self) -> None:
        """Test metadata with different tensor shapes."""
        shapes = [
            (1,),
            (10, 20),
            (3, 4, 5, 6),
            (1, 1, 1, 1, 1),
        ]
        for shape in shapes:
            metadata = TensorMetadata(
                shape=shape,
                dtype=TensorDtype.FLOAT32,
                source_framework="numpy",
                byte_size=100,
            )
            data = metadata.to_bytes()
            restored, _ = TensorMetadata.from_bytes(data)
            assert restored.shape == shape

    def test_metadata_all_dtypes(self) -> None:
        """Test metadata with all supported dtypes."""
        for dtype in TensorDtype:
            metadata = TensorMetadata(
                shape=(10,),
                dtype=dtype,
                source_framework="numpy",
                byte_size=40,
            )
            data = metadata.to_bytes()
            restored, _ = TensorMetadata.from_bytes(data)
            assert restored.dtype == dtype


class TestSerializedTensor:
    """Tests for SerializedTensor wire format."""

    def test_serialized_tensor_roundtrip(self) -> None:
        """Test that SerializedTensor can be serialized and deserialized."""
        metadata = TensorMetadata(
            shape=(2, 3),
            dtype=TensorDtype.FLOAT32,
            source_framework="numpy",
            byte_size=24,
        )
        tensor_data = b"\x00" * 24
        serialized = SerializedTensor(metadata=metadata, data=tensor_data)

        wire_bytes = serialized.to_bytes()
        restored = SerializedTensor.from_bytes(wire_bytes)

        assert restored.metadata.shape == metadata.shape
        assert restored.metadata.dtype == metadata.dtype
        assert restored.data == tensor_data


class TestTensorSerializerNumpy:
    """Tests for TensorSerializer with NumPy arrays."""

    def test_numpy_float32_roundtrip(self) -> None:
        """Test NumPy float32 array serialization roundtrip."""
        serializer = TensorSerializer()
        original = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

        serialized = serializer.serialize_numpy(original)
        restored = serializer.deserialize_to_numpy(serialized)

        np.testing.assert_array_equal(restored, original)
        assert serialized.metadata.dtype == TensorDtype.FLOAT32
        assert serialized.metadata.shape == (2, 3)
        assert serialized.metadata.source_framework == "numpy"

    def test_numpy_float16_roundtrip(self) -> None:
        """Test NumPy float16 array serialization roundtrip."""
        serializer = TensorSerializer()
        original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16)

        serialized = serializer.serialize_numpy(original)
        restored = serializer.deserialize_to_numpy(serialized)

        np.testing.assert_array_equal(restored, original)
        assert serialized.metadata.dtype == TensorDtype.FLOAT16

    def test_numpy_int_types(self) -> None:
        """Test NumPy integer type serialization."""
        serializer = TensorSerializer()
        int_types = [np.int8, np.int16, np.int32, np.int64, np.uint8]

        for int_type in int_types:
            original = np.array([1, 2, 3, 4, 5], dtype=int_type)
            serialized = serializer.serialize_numpy(original)
            restored = serializer.deserialize_to_numpy(serialized)
            np.testing.assert_array_equal(restored, original)

    def test_numpy_multidimensional(self) -> None:
        """Test NumPy multidimensional array serialization."""
        serializer = TensorSerializer()
        original = np.random.randn(2, 3, 4, 5).astype(np.float32)

        serialized = serializer.serialize_numpy(original)
        restored = serializer.deserialize_to_numpy(serialized)

        np.testing.assert_array_almost_equal(restored.astype(np.float64), original.astype(np.float64))
        assert serialized.metadata.shape == (2, 3, 4, 5)

    def test_numpy_non_contiguous(self) -> None:
        """Test that non-contiguous arrays are handled correctly."""
        serializer = TensorSerializer()
        # Create a non-contiguous array via slicing
        base = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
        original = base[::2, ::2]  # Non-contiguous view
        assert not original.flags["C_CONTIGUOUS"]

        serialized = serializer.serialize_numpy(original)
        restored = serializer.deserialize_to_numpy(serialized)

        np.testing.assert_array_equal(restored, original)


class TestTensorSerializerMlx:
    """Tests for TensorSerializer with MLX arrays."""

    def test_mlx_float32_roundtrip(self) -> None:
        """Test MLX float32 array serialization roundtrip."""
        import mlx.core as mx

        serializer = TensorSerializer()
        original = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=mx.float32)

        serialized = serializer.serialize_mlx(original)
        restored = serializer.deserialize_to_mlx(serialized)

        np.testing.assert_array_equal(np.array(restored), np.array(original))
        assert serialized.metadata.dtype == TensorDtype.FLOAT32
        assert serialized.metadata.shape == (2, 3)
        assert serialized.metadata.source_framework == "mlx"

    def test_mlx_float16_roundtrip(self) -> None:
        """Test MLX float16 array serialization roundtrip."""
        import mlx.core as mx

        serializer = TensorSerializer()
        original = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)

        serialized = serializer.serialize_mlx(original)
        restored = serializer.deserialize_to_mlx(serialized)

        np.testing.assert_array_almost_equal(
            np.array(restored), np.array(original), decimal=2
        )
        assert serialized.metadata.dtype == TensorDtype.FLOAT16

    def test_mlx_to_numpy_conversion(self) -> None:
        """Test MLX array to NumPy conversion via serialization."""
        import mlx.core as mx

        serializer = TensorSerializer()
        mlx_array = mx.array([[1, 2], [3, 4]], dtype=mx.int32)

        serialized = serializer.serialize_mlx(mlx_array)
        numpy_array = serializer.deserialize_to_numpy(serialized)

        np.testing.assert_array_equal(numpy_array, [[1, 2], [3, 4]])
        assert numpy_array.dtype == np.int32

    def test_numpy_to_mlx_conversion(self) -> None:
        """Test NumPy array to MLX conversion via serialization."""
        import mlx.core as mx

        serializer = TensorSerializer()
        numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        serialized = serializer.serialize_numpy(numpy_array)
        mlx_array = serializer.deserialize_to_mlx(serialized)
        # Verify it's an MLX array type
        assert isinstance(mlx_array, mx.array)

        np.testing.assert_array_equal(np.array(mlx_array), numpy_array)


class TestTensorSerializerGenericInterface:
    """Tests for the generic serialize/deserialize interface."""

    def test_serialize_dispatch_numpy(self) -> None:
        """Test that serialize correctly dispatches for numpy arrays."""
        serializer = TensorSerializer()
        array = np.array([1, 2, 3], dtype=np.float32)

        serialized = serializer.serialize(array, "numpy")
        assert serialized.metadata.source_framework == "numpy"

    def test_serialize_dispatch_mlx(self) -> None:
        """Test that serialize correctly dispatches for MLX arrays."""
        import mlx.core as mx

        serializer = TensorSerializer()
        array = mx.array([1, 2, 3], dtype=mx.float32)

        serialized = serializer.serialize(array, "mlx")
        assert serialized.metadata.source_framework == "mlx"

    def test_deserialize_dispatch_numpy(self) -> None:
        """Test that deserialize correctly dispatches to numpy."""
        serializer = TensorSerializer()
        array = np.array([1, 2, 3], dtype=np.float32)
        serialized = serializer.serialize_numpy(array)

        restored = serializer.deserialize(serialized, "numpy")
        assert isinstance(restored, np.ndarray)

    def test_deserialize_dispatch_mlx(self) -> None:
        """Test that deserialize correctly dispatches to MLX."""
        import mlx.core as mx

        serializer = TensorSerializer()
        array = np.array([1, 2, 3], dtype=np.float32)
        serialized = serializer.serialize_numpy(array)

        restored = serializer.deserialize(serialized, "mlx")
        assert isinstance(restored, mx.array)

    def test_cross_framework_mlx_to_numpy(self) -> None:
        """Test MLX -> bytes -> NumPy conversion."""
        import mlx.core as mx

        serializer = TensorSerializer()
        mlx_array = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.float32)

        # Serialize from MLX
        serialized = serializer.serialize(mlx_array, "mlx")

        # Deserialize to NumPy
        numpy_array = serializer.deserialize(serialized, "numpy")

        np.testing.assert_array_equal(numpy_array, [[1.0, 2.0], [3.0, 4.0]])

    def test_cross_framework_numpy_to_mlx(self) -> None:
        """Test NumPy -> bytes -> MLX conversion."""
        import mlx.core as mx

        serializer = TensorSerializer()
        numpy_array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

        # Serialize from NumPy
        serialized = serializer.serialize(numpy_array, "numpy")

        # Deserialize to MLX
        mlx_array = serializer.deserialize(serialized, "mlx")
        # Verify it's an MLX array type
        assert isinstance(mlx_array, mx.array)

        np.testing.assert_array_equal(np.array(mlx_array), numpy_array)


class TestTensorSerializerErrors:
    """Tests for error handling in TensorSerializer."""

    def test_serialize_invalid_type_numpy(self) -> None:
        """Test that serializing non-array as numpy raises TypeError."""
        serializer = TensorSerializer()

        with pytest.raises(TypeError, match="Expected np.ndarray"):
            serializer.serialize("not an array", "numpy")

    def test_serialize_invalid_type_mlx(self) -> None:
        """Test that serializing non-MLX array as mlx raises TypeError."""
        serializer = TensorSerializer()

        with pytest.raises(TypeError, match="Expected mx.array"):
            serializer.serialize_mlx("not an mlx array")

    def test_serialize_unsupported_framework(self) -> None:
        """Test that unsupported framework raises ValueError."""
        serializer = TensorSerializer()
        array = np.array([1, 2, 3])

        with pytest.raises(ValueError, match="Unsupported source framework"):
            serializer.serialize(array, "unsupported")  # type: ignore

    def test_deserialize_unsupported_framework(self) -> None:
        """Test that unsupported target framework raises ValueError."""
        serializer = TensorSerializer()
        array = np.array([1, 2, 3], dtype=np.float32)
        serialized = serializer.serialize_numpy(array)

        with pytest.raises(ValueError, match="Unsupported target framework"):
            serializer.deserialize(serialized, "unsupported")  # type: ignore

    def test_unsupported_numpy_dtype(self) -> None:
        """Test that unsupported numpy dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported numpy dtype"):
            TensorDtype.from_numpy_dtype(np.dtype("complex64"))


class TestTensorSerializerWireFormat:
    """Tests for the wire format used in network transfer."""

    def test_wire_format_size(self) -> None:
        """Test that wire format has expected size."""
        serializer = TensorSerializer()
        array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # 16 bytes

        serialized = serializer.serialize_numpy(array)
        wire_bytes = serialized.to_bytes()

        # 4 (metadata length) + metadata_size + 16 (data)
        # Metadata: 4 (num_dims) + 8 (2 dims) + 1 (dtype) + 1 (framework) + 8 (byte_size) = 22
        expected_metadata_size = 4 + 8 + 1 + 1 + 8  # 22 bytes
        expected_total = 4 + expected_metadata_size + 16
        assert len(wire_bytes) == expected_total

    def test_wire_format_roundtrip_full(self) -> None:
        """Test full wire format roundtrip including bytes conversion."""
        serializer = TensorSerializer()
        original = np.random.randn(10, 20).astype(np.float32)

        # Serialize
        serialized = serializer.serialize_numpy(original)
        wire_bytes = serialized.to_bytes()

        # Deserialize from bytes
        restored_serialized = SerializedTensor.from_bytes(wire_bytes)
        restored = serializer.deserialize_to_numpy(restored_serialized)

        np.testing.assert_array_almost_equal(restored.astype(np.float64), original.astype(np.float64))

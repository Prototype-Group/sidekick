import numpy as np
from PIL import Image

import pytest
from sidekick.encode import (OUTPUT_ENCODERS, CategoricalOutputEncoder,
                             FloatTensorEncoder, ImageEncoder, NumericEncoder,
                             NumpyEncoder, TextEncoder, TextOrIntInputEncoder,
                             get_encoder)


def test_numeric_encoder():
    encoder = NumericEncoder()

    encoder.check_type(2)
    encoder.check_type(2.1)

    with pytest.raises(TypeError):
        encoder.check_type("string")


def test_text_encoder():
    encoder = TextEncoder()
    value = "string"

    encoder.check_type(value)
    encoder.check_shape(value, shape=(100,))

    with pytest.raises(TypeError):
        encoder.check_type(34)


def test_text_or_int_encoder():
    encoder = TextOrIntInputEncoder()
    value = "string"

    encoder.check_type(value)
    encoder.check_shape(value, shape=(100,))

    encoder.check_type(34)

    with pytest.raises(TypeError):
        encoder.check_type(34.3)


def test_categorical__output_encoder():
    encoder = CategoricalOutputEncoder()
    value = {"a": 1, "b": 2}

    encoder.check_type(value)
    with pytest.raises(TypeError):
        encoder.check_type(5)

    encoder.check_shape(value, (2,))
    with pytest.raises(ValueError):
        encoder.check_shape(value, (3,))


def test_image_encoder():
    encoder = ImageEncoder()

    shape = (100, 10, 3)
    arr = np.uint8(np.random.rand(*shape) * 255)
    image = Image.fromarray(arr)

    encoder.check_type(image)
    with pytest.raises(TypeError):
        encoder.check_type(arr)

    with pytest.raises(ValueError):
        encoder.file_extension(image)

    image.format = "PNG"
    assert encoder.file_extension(image) == "png"
    assert encoder.media_type(image) == "image/png"

    encoder.check_shape(image, shape)


def test_floattensor_encoder():
    encoder = FloatTensorEncoder()

    shape = (100, 10, 3)
    arr = np.random.rand(*shape).astype(np.float32)

    encoder.check_shape(arr, shape)
    with pytest.raises(ValueError):
        encoder.check_shape(arr, (99, 10, 3))

    encoder.check_type(arr)
    with pytest.raises(TypeError):
        encoder.check_type([1, 2, 3])


def test_numpy_encoder():
    encoder = NumpyEncoder()

    shape = (100, 10, 3)
    arr = np.random.rand(*shape).astype(np.float32)

    encoder.check_shape(arr, shape)
    with pytest.raises(ValueError):
        encoder.check_shape(arr, (99, 10, 3))

    encoder.check_type(arr)
    with pytest.raises(TypeError):
        encoder.check_type([1, 2, 3])


def test_get_encoder():
    assert (
        get_encoder(
            dtype="numeric",
            shape=(1,),
            tensor_json=False,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["numeric"]
    )
    assert (
        get_encoder(
            dtype="numeric",
            shape=(2,),
            tensor_json=False,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["numpy"]
    )
    assert (
        get_encoder(
            dtype="numeric",
            shape=(2, 2),
            tensor_json=False,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["numpy"]
    )
    assert (
        get_encoder(
            dtype="image",
            shape=(2, 3),
            tensor_json=False,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["image"]
    )
    # Remove numpy output
    assert (
        get_encoder(
            dtype="numeric",
            shape=(1,),
            tensor_json=True,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["numeric"]
    )
    assert (
        get_encoder(
            dtype="numeric",
            shape=(2,),
            tensor_json=True,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["floattensor"]
    )
    assert (
        get_encoder(
            dtype="numeric",
            shape=(2, 2),
            tensor_json=True,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["floattensor"]
    )
    assert (
        get_encoder(
            dtype="image",
            shape=(2, 3),
            tensor_json=True,
            encoders=OUTPUT_ENCODERS,
        )
        is OUTPUT_ENCODERS["image"]
    )

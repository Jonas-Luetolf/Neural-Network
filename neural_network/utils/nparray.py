import numpy as np


def to_categorical(y: np.ndarray) -> np.ndarray:
    y = np.array(y, dtype="int")
    input_shape = y.shape
    dtype = "float32"

    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)

    num_classes = np.max(y) + 1
    n = y.shape[0]

    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1

    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

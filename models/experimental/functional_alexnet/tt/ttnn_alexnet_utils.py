# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def preprocess_linear_parameter(device, path, state_dict, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    weight = state_dict[f"{path}.weight"]
    bias = state_dict[f"{path}.bias"]

    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, device=device)

    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, device=device)
    return (weight, bias)


def preprocess_conv_parameter(device, path, state_dict, dtype=ttnn.float32):
    conv_weight = state_dict[f"{path}.weight"]
    conv_bias = state_dict[f"{path}.bias"]

    conv_weight = ttnn.from_torch(conv_weight, dtype=dtype)
    conv_bias = ttnn.from_torch(conv_bias.reshape((1, 1, 1, -1)), dtype=dtype)

    return (conv_weight, conv_bias)


def custom_preprocessor(device, state_dict):
    pairs = [
        ("features.0", "conv"),
        ("features.3", "conv"),
        ("features.6", "conv"),
        ("features.8", "conv"),
        ("features.10", "conv"),
        ("classifier.1", "linear"),
        ("classifier.4", "linear"),
        ("classifier.6", "linear"),
    ]

    parameters = {}

    for path, layer in pairs:
        if layer == "conv":
            parameters[path] = preprocess_conv_parameter(device, path, state_dict, dtype=ttnn.bfloat16)
        else:
            parameters[path] = preprocess_linear_parameter(device, path, state_dict, dtype=ttnn.bfloat16)

    return parameters

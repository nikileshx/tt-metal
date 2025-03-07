# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, bfloat8=False):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    if bfloat8:
        return (ttnn.from_torch(weight, dtype=ttnn.float32), ttnn.from_torch(bias, dtype=ttnn.float32))

    return (ttnn.from_torch(weight, dtype=ttnn.bfloat16), ttnn.from_torch(bias, dtype=ttnn.bfloat16))


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p if isinstance(p, int) else p[0]


def make_anchors(device, feats, strides, grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i]
        sx = torch.arange(end=w) + grid_cell_offset
        sy = torch.arange(end=h) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    a = torch.cat(anchor_points).transpose(0, 1).unsqueeze(0)
    b = torch.cat(stride_tensor).transpose(0, 1)

    return (
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def ttnn_decode_bboxes(device, distance, anchor_points, xywh=True):
    distance = ttnn.to_layout(distance, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    # lt, rb = ttnn.split(distance, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)  # if done in tile : tt-metal issue #17017

    a, b, c = distance.shape
    lt = ttnn.slice(distance, [0, 0, 0], [a, b // 2, c], memory_config=ttnn.L1_MEMORY_CONFIG)
    rb = ttnn.slice(distance, [0, b // 2, 0], [a, b, c], memory_config=ttnn.L1_MEMORY_CONFIG)

    lt = ttnn.to_layout(lt, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    rb = ttnn.to_layout(rb, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    x1y1 = ttnn.subtract(anchor_points, lt, memory_config=ttnn.L1_MEMORY_CONFIG)
    x2y2 = ttnn.add(anchor_points, rb, memory_config=ttnn.L1_MEMORY_CONFIG)

    if xywh:
        c_xy = ttnn.add(x1y1, x2y2, memory_config=ttnn.L1_MEMORY_CONFIG)
        c_xy = ttnn.div(c_xy, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        wh = ttnn.subtract(x2y2, x1y1, memory_config=ttnn.L1_MEMORY_CONFIG)

        # Concat in tile with bs 8 is throwing error
        c_xy = ttnn.to_layout(c_xy, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        wh = ttnn.to_layout(wh, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

        result = ttnn.concat((c_xy, wh), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        result = ttnn.to_layout(result, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        return result


def preprocess_parameters(state_dict, path, bias=True, bfloat8=True):
    if bias:
        conv_weight = state_dict[f"{path}.2.weight"]
        conv_bias = state_dict[f"{path}.2.bias"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.float32), (1, 1, 1, -1))
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16), (1, 1, 1, -1))

        return (conv_weight, conv_bias)

    else:
        conv_weight = state_dict[f"{path}.conv.weight"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)

        return (conv_weight, None)


def custom_preprocessor(device, state_dict, inp_h=320, inp_w=320):
    pairs = [
        ("model.0", True),
        ("model.1", True),
        ("model.2.cv1", True),
        ("model.2.m.0.cv1", True),
        ("model.2.m.0.cv2", True),
        ("model.2.m.1.cv1", True),
        ("model.2.m.1.cv2", True),
        ("model.2.cv2", True),
        ("model.3", True),
        ("model.4.cv1", True),
        ("model.4.m.0.cv1", True),
        ("model.4.m.0.cv2", True),
        ("model.4.m.1.cv1", True),
        ("model.4.m.1.cv2", True),
        ("model.4.m.2.cv1", True),
        ("model.4.m.2.cv2", True),
        ("model.4.m.3.cv1", True),
        ("model.4.m.3.cv2", True),
        ("model.4.cv2", True),
        ("model.5", True),
        ("model.6.cv1", True),
        ("model.6.m.0.cv1", True),
        ("model.6.m.0.cv2", True),
        ("model.6.m.1.cv1", True),
        ("model.6.m.1.cv2", True),
        ("model.6.m.2.cv1", True),
        ("model.6.m.2.cv2", True),
        ("model.6.m.3.cv1", True),
        ("model.6.m.3.cv2", True),
        ("model.6.cv2", True),
        ("model.7", True),
        ("model.8.cv1", True),
        ("model.8.m.0.cv1", True),
        ("model.8.m.0.cv2", True),
        ("model.8.m.1.cv1", True),
        ("model.8.m.1.cv2", True),
        ("model.8.cv2", True),
        ("model.9.cv1", True),
        ("model.9.cv2", True),
        ("model.12.cv1", True),
        ("model.12.m.0.cv1", True),
        ("model.12.m.0.cv2", True),
        ("model.12.m.1.cv1", True),
        ("model.12.m.1.cv2", True),
        ("model.12.cv2", True),
        ("model.15.cv1", True),
        ("model.15.m.0.cv1", True),
        ("model.15.m.0.cv2", True),
        ("model.15.m.1.cv1", True),
        ("model.15.m.1.cv2", True),
        ("model.15.cv2", True),
        ("model.16", True),
        ("model.18.cv1", True),
        ("model.18.m.0.cv1", True),
        ("model.18.m.0.cv2", True),
        ("model.18.m.1.cv1", True),
        ("model.18.m.1.cv2", True),
        ("model.18.cv2", True),
        ("model.19", True),
        ("model.21.cv1", True),
        ("model.21.m.0.cv1", True),
        ("model.21.m.0.cv2", True),
        ("model.21.m.1.cv1", True),
        ("model.21.m.1.cv2", True),
        ("model.21.cv2", True),
        ("model.22.cv2.0.0", True),
        ("model.22.cv2.0.1", True),
        ("model.22.cv3.0.0", True),
        ("model.22.cv3.0.1", True),
        ("model.22.cv2.1.0", True),
        ("model.22.cv2.1.1", True),
        ("model.22.cv3.1.0", True),
        ("model.22.cv3.1.1", True),
        ("model.22.cv2.2.0", True),
        ("model.22.cv2.2.1", True),
        ("model.22.cv3.2.0", True),
        ("model.22.cv3.2.1", True),
    ]

    parameters = {}

    for path, bfloat8 in pairs:
        parameters[path] = fold_batch_norm2d_into_conv2d(device, state_dict, path=path, bfloat8=bfloat8)

    # Detect

    parameters["model.22.cv2.0"] = preprocess_parameters(state_dict, "model.22.cv2.0")
    parameters["model.22.cv3.0"] = preprocess_parameters(state_dict, "model.22.cv3.0")

    parameters["model.22.cv2.1"] = preprocess_parameters(state_dict, "model.22.cv2.1")
    parameters["model.22.cv3.1"] = preprocess_parameters(state_dict, "model.22.cv3.1")

    parameters["model.22.cv2.2"] = preprocess_parameters(state_dict, "model.22.cv2.2")
    parameters["model.22.cv3.2"] = preprocess_parameters(state_dict, "model.22.cv3.2")

    # DFL conv

    parameters["model.22.dfl"] = preprocess_parameters(state_dict, "model.22.dfl", bias=False)

    strides = [8, 16, 32]

    feats = [(inp_h // 8, inp_w // 8), (inp_h // 16, inp_w // 16), (inp_h // 32, inp_w // 32)]  # value depend on res

    anchors, strides = make_anchors(device, feats, strides)  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    return parameters


def get_concat_shard(device, input_tensors, num_cores=64, dim=3):
    input_sharded_memory_config = []

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    for i in range(len(input_tensors)):
        in_shard_width = input_tensors[i].shape[-1]
        shard_height = (input_tensors[i].shape[2] + num_cores - 1) // num_cores
        memory_config = ttnn.create_sharded_memory_config(
            (shard_height, in_shard_width),
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_config.append(memory_config)

    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config[i])

    # in_shard_width = input_tensors[1].shape[-1]
    # shard_height = (input_tensors[1].shape[2] + num_cores - 1) // num_cores
    # memory_config = ttnn.create_sharded_memory_config(
    #     (shard_height, in_shard_width),
    #     core_grid=shard_grid,
    #     strategy=ttnn.ShardStrategy.HEIGHT,
    #     use_height_and_width_as_shard_shape=True,
    # )
    # input_sharded_memory_config.append(memory_config)

    # input_tensors[1] = ttnn.to_memory_config(input_tensors[1], input_sharded_memory_config[0])

    # out_shard_width = 0
    # for i in range(len(input_tensors)):
    #     out_shard_width += input_tensors[i].shape[-1]

    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    output = ttnn.sharded_to_interleaved(output, ttnn.L1_MEMORY_CONFIG)

    return output

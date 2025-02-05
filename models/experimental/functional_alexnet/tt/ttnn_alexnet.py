# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


def ttnn_alexnet(device, x, parameters):
    batch_size = x.shape[0]

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv1_weight = ttnn.from_device(parameters.features[0].weight)
    conv1_bias = ttnn.from_device(parameters.features[0].bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv1_weight,
        in_channels=3,
        out_channels=64,
        device=device,
        bias_tensor=conv1_bias,
        kernel_size=(11, 11),
        stride=(4, 4),
        padding=(2, 2),
        batch_size=batch_size,
        input_height=x.shape[1],
        input_width=x.shape[2],
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        memory_config=None,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    conv2_weight = ttnn.from_device(parameters.features[3].weight)
    conv2_bias = ttnn.from_device(parameters.features[3].bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv2_weight,
        in_channels=64,
        out_channels=192,
        device=device,
        bias_tensor=conv2_bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        batch_size=batch_size,
        input_height=27,
        input_width=27,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    conv3_weight = ttnn.from_device(parameters.features[6].weight)
    conv3_bias = ttnn.from_device(parameters.features[6].bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv3_weight,
        in_channels=192,
        out_channels=384,
        device=device,
        bias_tensor=conv3_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=13,
        input_width=13,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    conv4_weight = ttnn.from_device(parameters.features[8].weight)
    conv4_bias = ttnn.from_device(parameters.features[8].bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv4_weight,
        in_channels=384,
        out_channels=256,
        device=device,
        bias_tensor=conv4_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    conv5_weight = ttnn.from_device(parameters.features[10].weight)
    conv5_bias = ttnn.from_device(parameters.features[10].bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv5_weight,
        in_channels=256,
        out_channels=256,
        device=device,
        bias_tensor=conv5_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    x = ttnn.reshape(x, (batch_size, 6, 6, 256), memory_config=ttnn.L1_MEMORY_CONFIG)

    # ttnn currently only support AAP2 with output_size=(1,1), so torch op has been used.

    avg_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

    tt_output_tensor = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    x = avg_pool(torch_output_tensor)

    x = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    x = ttnn.reshape(x, (x.shape[0], -1), memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.linear(
        x,
        parameters.classifier[1].weight,
        bias=parameters.classifier[1].bias,
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    x = ttnn.linear(
        x,
        parameters.classifier[4].weight,
        bias=parameters.classifier[4].bias,
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.linear(
        x, parameters.classifier[6].weight, bias=parameters.classifier[6].bias, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    return x

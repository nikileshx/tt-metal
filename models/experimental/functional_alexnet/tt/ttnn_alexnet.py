# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def Conv(
    device, x, parameters, path, inp_h, inp_w, k, s, p, width_shard=False, output_layout=ttnn.TILE_LAYOUT, batch_size=1
):
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if False or (x.shape[3] == 16 and x.shape[-2] == 115) else 32),
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if width_shard:
        conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_weight, conv_bias = parameters[path]

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv_weight,
        in_channels=conv_weight.shape[1],
        out_channels=conv_weight.shape[0],
        device=device,
        bias_tensor=conv_bias,
        kernel_size=(k, k),
        stride=(s, s),
        padding=(p, p),
        batch_size=batch_size,
        input_height=inp_h,
        input_width=inp_w,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        memory_config=None,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    return x, out_height, out_width


def ttnn_alexnet(device, x, parameters):
    batch_size = x.shape[0]

    x, out_height, out_width = Conv(
        device, x, parameters, "features.0", inp_h=x.shape[1], inp_w=x.shape[2], k=11, s=4, p=2, batch_size=batch_size
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

    x, out_height, out_width = Conv(
        device, x, parameters, "features.3", inp_h=27, inp_w=27, k=5, s=1, p=2, batch_size=batch_size
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

    x, out_height, out_width = Conv(
        device, x, parameters, "features.6", inp_h=13, inp_w=13, k=3, s=1, p=1, width_shard=True, batch_size=batch_size
    )

    x, out_height, out_width = Conv(
        device,
        x,
        parameters,
        "features.8",
        inp_h=out_height,
        inp_w=out_width,
        k=3,
        s=1,
        p=1,
        width_shard=True,
        batch_size=batch_size,
    )

    x, out_height, out_width = Conv(
        device,
        x,
        parameters,
        "features.10",
        inp_h=out_height,
        inp_w=out_width,
        k=3,
        s=1,
        p=1,
        width_shard=True,
        batch_size=batch_size,
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

    x = ttnn.adaptive_avg_pool2d(x, ttnn.Shape([6, 6]))

    x = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.reshape(x, (batch_size, -1), memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.linear(
        x,
        parameters["classifier.1"][0],
        bias=parameters["classifier.1"][1],
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    x = ttnn.linear(
        x,
        parameters["classifier.4"][0],
        bias=parameters["classifier.4"][1],
        activation="relu",
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    x = ttnn.linear(
        x, parameters["classifier.6"][0], bias=parameters["classifier.6"][1], memory_config=ttnn.L1_MEMORY_CONFIG
    )

    return x

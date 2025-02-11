# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def Conv(
    device,
    x,
    parameters,
    path,
    inp_h,
    inp_w,
    k,
    s,
    p,
    width_shard=False,
    output_layout=ttnn.TILE_LAYOUT,
    batch_size=1,
    memory_config=None,
    change_shard=False,
    c1=None,
    c2=None,
):
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=16 if x.shape[3] < 16 else 32,
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
    if change_shard:
        conv_config.shard_layout = None

    conv_kwargs = {
        "in_channels": c1,
        "out_channels": c2,
        "batch_size": x.shape[0],
        "input_height": inp_h,
        "input_width": inp_w,
        "kernel_size": (k, k),
        "stride": (s, s),
        "padding": (p, p),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    input_memory_config = (
        ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG if True and not width_shard else ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
    )

    conv_weight, conv_bias = parameters[path]

    if not ttnn.is_tensor_storage_on_device(conv_weight):
        conv_weight = ttnn.prepare_conv_weights(
            weight_tensor=conv_weight,
            weights_format="OIHW",
            input_memory_config=input_memory_config,
            input_layout=ttnn.TILE_LAYOUT,
            has_bias=True,
            **conv_kwargs,
        )

        conv_bias = ttnn.prepare_conv_bias(
            bias_tensor=conv_bias,
            input_memory_config=input_memory_config,
            input_layout=ttnn.TILE_LAYOUT,
            **conv_kwargs,
        )
        conv_weight = ttnn.to_device(conv_weight, device)
        conv_bias = ttnn.to_device(conv_bias, device)

        parameters[path] = (conv_weight, conv_bias)

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv_weight,
        in_channels=c1,
        out_channels=c2,
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
        memory_config=memory_config,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    return x, out_height, out_width


def ttnn_alexnet(device, x, parameters):
    batch_size = x.shape[0]

    x, out_height, out_width = Conv(
        device, x, parameters, "features.0", inp_h=224, inp_w=224, k=11, s=4, p=2, batch_size=batch_size, c1=3, c2=64
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
        device, x, parameters, "features.3", inp_h=27, inp_w=27, k=5, s=1, p=2, batch_size=batch_size, c1=64, c2=192
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
        device,
        x,
        parameters,
        "features.6",
        inp_h=13,
        inp_w=13,
        k=3,
        s=1,
        p=1,
        width_shard=True,
        batch_size=batch_size,
        c1=192,
        c2=384,
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
        c1=384,
        c2=256,
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
        c1=256,
        c2=256,
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
        x,
        parameters["classifier.6"][0],
        bias=parameters["classifier.6"][1],
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return x

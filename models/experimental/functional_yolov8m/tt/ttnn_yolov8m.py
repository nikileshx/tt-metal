# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math

from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import (
    autopad,
    ttnn_decode_bboxes,
)


def conv(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    k=1,
    s=1,
    p=None,
    g=1,
    d=1,
    act_block_h=False,
    block_shard=None,
    bfloat8=True,
    change_shard=False,
    deallocate_activation=False,
    output_layout=ttnn.TILE_LAYOUT,
    is_fused=True,
    is_dfl=False,
    is_detect_cv2=False,
    width_shard=False,
    memory_config=None,
    batch_size=1,
):
    p = autopad(k, p, d)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=16 if x.shape[3] < 16 else 32,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )

    if deallocate_activation:
        conv_config.deallocate_activation = deallocate_activation

    if change_shard:
        conv_config.shard_layout = None

    if act_block_h:
        conv_config.act_block_h_override = 32

    if block_shard:
        conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    if width_shard:
        conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if is_fused:
        fused_weight, fused_bias = parameters[path]
    else:
        conv_weight, conv_bias = parameters[path]

    if bfloat8:
        conv_config.weights_dtype = ttnn.bfloat8_b

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=fused_weight if is_fused else conv_weight,
        in_channels=fused_weight.shape[1] if is_fused else conv_weight.shape[1],
        out_channels=fused_weight.shape[0] if is_fused else conv_weight.shape[0],
        device=device,
        bias_tensor=fused_bias if is_fused else conv_bias,
        kernel_size=(k, k),
        stride=(s, s),
        padding=(p, p),
        dilation=(d, d),
        batch_size=batch_size,
        input_height=in_h,
        input_width=in_w,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=g,
        memory_config=memory_config,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    if is_dfl:
        return ttnn.reshape(x, (batch_size, 4, -1), memory_config=ttnn.L1_MEMORY_CONFIG)

    if is_detect_cv2:
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return (x, out_height, out_width)

    x = ttnn.silu(x)

    return (x, out_height, out_width)


def Bottleneck(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    shortcut=True,
    g=1,
    k=(3, 3),
    e=0.5,
    act_block_h=True,
    change_shard=None,
    deallocate_activation=False,
    output_layout=ttnn.TILE_LAYOUT,
    tilize=False,
    batch_size=1,
    temp="",
    width_shard=False,
    block_shard=False,
):
    cv1, out_h, out_w = conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        in_h,
        in_w,
        k[0][0],
        1,
        act_block_h=act_block_h,
        change_shard=change_shard,
        deallocate_activation=deallocate_activation,
        output_layout=output_layout,
        batch_size=batch_size,
        width_shard=width_shard,
        block_shard=block_shard,
    )

    if temp == "c2f8":
        cv1 = ttnn.from_device(cv1)

    cv2, out_h, out_w = conv(
        device,
        cv1,
        parameters,
        f"{path}.cv2",
        in_h,
        in_w,
        k[1][1],
        1,
        g=g,
        act_block_h=act_block_h,
        deallocate_activation=deallocate_activation,
        batch_size=batch_size,
        width_shard=width_shard,
        block_shard=block_shard,
    )

    ttnn.deallocate(cv1)

    if tilize:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    add = shortcut

    return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if add else cv2


def c2f(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    n=1,
    shortcut=False,
    g=1,
    e=0.5,
    act_block_h=False,
    bfloat8=True,
    block_shard=False,
    width_shard=False,
    change_shard=None,
    use_interleaved=False,
    batch_size=1,
    temp="",
):
    cv1, out_h, out_w = conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        in_h,
        in_w,
        1,
        1,
        bfloat8=bfloat8,
        change_shard=change_shard,
        width_shard=width_shard,
        batch_size=batch_size,
        block_shard=False if temp == "c2f18" else block_shard,
    )

    if use_interleaved:
        cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)

    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    if temp in ["c2f6", "c2f12", "c2f18", "c2f21"]:
        y = list(ttnn.split(cv1, 2, 3, memory_config=ttnn.L1_MEMORY_CONFIG))
    else:
        y = []
        bs, h, w, c = cv1.shape
        y.append(ttnn.slice(cv1, [0, 0, 0, 0], [bs, h, w, c // 2], memory_config=ttnn.L1_MEMORY_CONFIG))
        y.append(ttnn.slice(cv1, [0, 0, 0, c // 2], [bs, h, w, c], memory_config=ttnn.L1_MEMORY_CONFIG))
        # y.append(cv1[:, :, :, : cv1.shape[-1] // 2])
        # y.append(cv1[:, :, :, cv1.shape[-1] // 2 :])

    ttnn.deallocate(cv1)

    to_tile = True

    for i in range(n):
        z = Bottleneck(
            device,
            y[-1],
            parameters,
            f"{path}.m.{i}",
            out_h,
            out_w,
            shortcut,
            g,
            k=((3, 3), (3, 3)),
            e=1.0,
            act_block_h=act_block_h,
            change_shard=change_shard,
            tilize=to_tile,
            batch_size=batch_size,
            temp=temp,
            width_shard=width_shard,
            block_shard=block_shard,
        )
        y.append(z)
        to_tile = False

    y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    if not shortcut:
        for i in range(2, len(y)):
            y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)

    if batch_size > 6:
        x = ttnn.concat(y, 3)
    else:
        x = ttnn.concat(y, 3, memory_config=ttnn.L1_MEMORY_CONFIG)

    for i in range(len(y)):
        ttnn.deallocate(y[i])

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        f"{path}.cv2",
        out_h,
        out_w,
        1,
        bfloat8=bfloat8,
        block_shard=block_shard,
        change_shard=True if temp == "c2f8" else change_shard,
        batch_size=batch_size,
    )
    return x, out_h, out_w


def SPPF(device, x, parameters, path, in_h, in_w, k=5, batch_size=1):
    cv1, out_h, out_w = conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        in_h,
        in_w,
        1,
        1,
        change_shard=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        batch_size=batch_size,
    )

    p = k // 2

    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    y = [cv1]
    for i in range(3):
        output = ttnn.max_pool2d(
            input_tensor=y[-1],
            batch_size=batch_size,
            input_h=out_h,
            input_w=out_w,
            channels=y[-1].shape[-1],
            kernel_size=[5, 5],
            stride=[1, 1],
            padding=[p, p],
            dilation=[1, 1],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        y.append(output)

    x = ttnn.concat(y, 3, memory_config=ttnn.L1_MEMORY_CONFIG)

    for i in range(len(y)):
        ttnn.deallocate(y[i])

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        f"{path}.cv2",
        out_h,
        out_w,
        1,
        1,
        change_shard=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        batch_size=batch_size,
    )

    return x, out_h, out_w


def Detect_cv2(device, x, parameters, path, inp_h, inp_w, k, reg_max, bfloat8=True, batch_size=1):
    x, out_h, out_w = conv(device, x, parameters, f"{path}.0", inp_h, inp_w, k, bfloat8=bfloat8, batch_size=batch_size)

    x, out_h, out_w = conv(device, x, parameters, f"{path}.1", out_h, out_w, k, bfloat8=bfloat8, batch_size=batch_size)

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        path,
        out_h,
        out_w,
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        is_fused=False,
        change_shard=True,
        is_detect_cv2=True,
        batch_size=batch_size,
    )
    return x, out_h, out_w


def DFL(device, x, parameters, path, c1=16, batch_size=1):
    b, _, a = x.shape

    x = ttnn.reshape(x, (b, 4, c1, a), memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.softmax(x, dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.permute(x, (0, 1, 3, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

    x = conv(
        device,
        x,
        parameters,
        path,
        x.shape[1],
        x.shape[2],
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        is_fused=False,
        is_dfl=True,
        change_shard=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        batch_size=batch_size,
    )

    return x


def Detect(device, x, parameters, path, nc=80, ch=(), bfloat8=True, batch_size=1):
    nc = nc
    nl = len(ch)
    reg_max = 16
    no = nc + reg_max * 4

    for i in range(nl):
        inp_h = inp_w = int(math.sqrt((x[i].shape[2]) // batch_size))
        a = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv2.{i}",
            inp_h=inp_h,
            inp_w=inp_w,
            k=3,
            reg_max=4 * reg_max,
            batch_size=batch_size,
        )[0]
        b = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv3.{i}",
            inp_h=inp_h,
            inp_w=inp_w,
            k=3,
            reg_max=nc,
            bfloat8=bfloat8,
            batch_size=batch_size,
        )[0]
        x[i] = ttnn.concat((a, b), dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

    anchors, strides = parameters["anchors"], parameters["strides"]

    xi = []
    for i in x:
        i = ttnn.reshape(i, (batch_size, -1, no), memory_config=ttnn.L1_MEMORY_CONFIG)
        xi.append(i)

    x_cat = ttnn.concat(xi, 1, memory_config=ttnn.L1_MEMORY_CONFIG)

    x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

    box = ttnn.slice(x_cat, [0, 0, 0], [batch_size, 64, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)
    cls = ttnn.slice(x_cat, [0, 64, 0], [batch_size, 144, x_cat.shape[2]], memory_config=ttnn.L1_MEMORY_CONFIG)

    dfl = DFL(device, box, parameters, f"{path}.dfl", batch_size=batch_size)

    dbox = ttnn_decode_bboxes(device, dfl, anchors)
    dbox = ttnn.multiply(dbox, strides, memory_config=ttnn.L1_MEMORY_CONFIG)

    return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]  # oup memory config as ttnn.L1 affects bounding boxes


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


def DetectionModel(device, x, parameters, res, batch_size):
    print(f"Inference running for batch size: {batch_size}")
    x, out_h, out_w = conv(
        device, x, parameters, "model.0", res[0], res[1], 3, 2, 1, act_block_h=True, batch_size=batch_size
    )

    if batch_size > 6:
        x = ttnn.from_device(x)

    x, out_h, out_w = conv(
        device, x, parameters, "model.1", out_h, out_w, 3, 2, 1, act_block_h=True, batch_size=batch_size
    )

    x, out_h, out_w = c2f(
        device, x, parameters, "model.2", out_h, out_w, n=2, shortcut=True, act_block_h=True, batch_size=batch_size
    )

    x, out_h, out_w = conv(device, x, parameters, "model.3", out_h, out_w, 3, 2, batch_size=batch_size)

    x, out_h, out_w = c2f(device, x, parameters, "model.4", out_h, out_w, n=4, shortcut=True, batch_size=batch_size)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    four = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        "model.5",
        out_h,
        out_w,
        3,
        2,
        1,
        change_shard=False,
        block_shard=False,
        batch_size=batch_size,
    )

    x, out_h, out_w = c2f(
        device,
        x,
        parameters,
        "model.6",
        out_h,
        out_w,
        n=4,
        shortcut=True,
        block_shard=True,
        change_shard=False,
        batch_size=batch_size,
        temp="c2f6",
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    six = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        "model.7",
        out_h,
        out_w,
        3,
        2,
        1,
        block_shard=True,
        change_shard=False,
        batch_size=batch_size,
    )

    x, out_h, out_w = c2f(
        device,
        x,
        parameters,
        "model.8",
        out_h,
        out_w,
        n=2,
        shortcut=True,
        block_shard=True,
        use_interleaved=True,
        batch_size=batch_size,
        temp="c2f8" if batch_size > 6 else "",
    )

    x, out_h, out_w = SPPF(device, x, parameters, "model.9", out_h, out_w, batch_size=batch_size)

    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    nine = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.reshape(x, (batch_size, out_h, out_w, x.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.upsample(x, scale_factor=(2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

    inp_h, inp_w = x.shape[1], x.shape[2]

    x = ttnn.reshape(x, (1, 1, batch_size * inp_h * inp_w, x.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)
    # x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # x = ttnn.concat([x, six], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = get_concat_shard(device, [x, six], dim=3)

    ttnn.deallocate(six)

    x, out_h, out_w = c2f(
        device,
        x,
        parameters,
        "model.12",
        inp_h,
        inp_w,
        n=2,
        shortcut=False,
        bfloat8=True,
        batch_size=batch_size,
        temp="c2f12",
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    twelve = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = ttnn.reshape(x, (batch_size, out_h, out_w, x.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.upsample(x, scale_factor=(2, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

    inp_h, inp_w = x.shape[1], x.shape[2]

    x = ttnn.reshape(x, (1, 1, batch_size * inp_h * inp_w, x.shape[-1]), memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
    x = ttnn.concat([x, four], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(four)

    x, out_h, out_w = c2f(device, x, parameters, "model.15", inp_h, inp_w, n=2, shortcut=False, batch_size=batch_size)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    fifteen = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        "model.16",
        out_h,
        out_w,
        3,
        2,
        1,
        batch_size=batch_size,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    # x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)

    # x = ttnn.concat([x, twelve], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

    # change

    x = get_concat_shard(device, [x, twelve], dim=3)

    ttnn.deallocate(twelve)

    x, out_h, out_w = c2f(
        device,
        x,
        parameters,
        "model.18",
        out_h,
        out_w,
        n=2,
        shortcut=False,
        batch_size=batch_size,
        temp="c2f18",
        block_shard=True,
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    eighteen = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)

    x, out_h, out_w = conv(
        device,
        x,
        parameters,
        "model.19",
        out_h,
        out_w,
        3,
        2,
        1,
        block_shard=True,
        batch_size=batch_size,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    # x = ttnn.concat([x, nine], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

    x = get_concat_shard(device, [x, nine], dim=3)

    ttnn.deallocate(nine)

    x, out_h, out_w = c2f(
        device, x, parameters, "model.21", out_h, out_w, n=2, shortcut=False, batch_size=batch_size, temp="c2f21"
    )

    x = [fifteen, eighteen, x]

    x = Detect(device, x, parameters, "model.22", nc=80, ch=(192, 384, 576), batch_size=batch_size)

    return x


def YOLOv8m(device, x, parameters, res=(320, 320), batch_size=1):
    return DetectionModel(device, x, parameters, res, batch_size=batch_size)

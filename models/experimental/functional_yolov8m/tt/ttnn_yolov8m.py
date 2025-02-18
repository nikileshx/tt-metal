# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import ttnn

from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import (
    autopad,
    ttnn_decode_bboxes,
)


def Conv(
    device,
    x,
    parameters,
    path,
    c1,
    c2,
    k=1,
    s=1,
    p=None,
    g=1,
    d=1,
    act_block_h=False,
    block_shard=None,
    bfloat8=True,
    change_shard=False,
    inp_h=None,
    inp_w=None,
    is_fused=True,
    is_dfl=False,
    width_shard=False,
    deallocate_activation=False,
    memory_config=None,
):
    p = autopad(k, p, d)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if False or (c1 == 16 and x.shape[-2] == 115) else 32),
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    if change_shard:
        conv_config.shard_layout = None

    if act_block_h:
        conv_config.act_block_h_override = 32

    if block_shard:
        conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    if deallocate_activation:
        conv_config.deallocate_activation = deallocate_activation

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
        fused_weight, fused_bias = parameters[path]  # no conv-batch fuse

    if bfloat8:
        conv_config.weights_dtype = ttnn.bfloat8_b

    [x, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=fused_weight,
        in_channels=c1,
        out_channels=c2,
        device=device,
        bias_tensor=fused_bias,
        kernel_size=(k, k),
        stride=(s, s),
        padding=(p, p),
        dilation=(d, d),
        batch_size=x.shape[0],
        input_height=inp_h,
        input_width=inp_w,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=g,
        memory_config=memory_config,
        return_weights_and_bias=True,
        return_output_dim=True,
    )

    if is_dfl:
        x = ttnn.reshape(x, (x.shape[0], 4, -1))
        return x

    x = ttnn.silu(x)

    return (x, out_height, out_width)


def Bottleneck(
    device,
    x,
    parameters,
    path,
    c1,
    c2,
    shortcut=True,
    g=1,
    k=(3, 3),
    e=0.5,
    act_block_h=True,
    change_shard=None,
    inp_h=None,
    inp_w=None,
    tilize=False,
):
    c_ = int(c2 * e)

    cv1, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        c1,
        c_,
        k[0][0],
        1,
        act_block_h=act_block_h,
        change_shard=change_shard,
        inp_h=inp_h,
        inp_w=inp_w,
    )

    cv2, out_h, out_w = Conv(
        device,
        cv1,
        parameters,
        f"{path}.cv2",
        c_,
        c2,
        k[1][1],
        1,
        g=g,
        act_block_h=act_block_h,
        change_shard=change_shard,
        inp_h=out_h,
        inp_w=out_w,
    )

    ttnn.deallocate(cv1)

    if tilize:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)

    add = shortcut

    return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if add else cv2


def C2f(
    device,
    x,
    parameters,
    path,
    c1,
    c2,
    n=1,
    shortcut=False,
    g=1,
    e=0.5,
    act_block_h=False,
    bfloat8=True,
    block_shard=False,
    change_shard=None,
    inp_h=None,
    inp_w=None,
):
    c = int(c2 * e)

    cv1, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        c1,
        2 * c,
        1,
        1,
        bfloat8=bfloat8,
        change_shard=change_shard,
        inp_h=inp_h,
        inp_w=inp_w,
    )

    cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)
    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
    y = list(ttnn.split(cv1, 2, 3))
    ttnn.deallocate(cv1)

    to_tile = True

    for i in range(n):
        z = Bottleneck(
            device,
            y[-1],
            parameters,
            f"{path}.m.{i}",
            c,
            c,
            shortcut,
            g,
            k=((3, 3), (3, 3)),
            e=1.0,
            act_block_h=act_block_h,
            change_shard=change_shard,
            inp_h=out_h,
            inp_w=out_w,
            tilize=to_tile,
        )
        y.append(z)
        to_tile = False

    y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT)
    y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT)

    if not shortcut:
        for i in range(2, len(y)):
            y[i] = ttnn.sharded_to_interleaved(y[i], ttnn.L1_MEMORY_CONFIG)

    x = ttnn.concat(y, 3)

    for i in range(len(y)):
        ttnn.deallocate(y[i])

    x, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv2",
        (2 + n) * c,
        c2,
        1,
        bfloat8=bfloat8,
        block_shard=block_shard,
        change_shard=change_shard,
        inp_h=out_h,
        inp_w=out_w,
    )
    return x, out_h, out_w


def SPPF(device, x, parameters, path, c1, c2, k=5, bfloat8=True, inp_h=None, inp_w=None):
    c_ = c1 // 2
    cv1, out_h, out_w = Conv(
        device, x, parameters, f"{path}.cv1", c1, c_, 1, 1, inp_h=inp_h, inp_w=inp_w, change_shard=True
    )

    p = k // 2
    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)

    y = [cv1]
    for i in range(3):
        output = ttnn.max_pool2d(
            input_tensor=y[-1],
            batch_size=y[-1].shape[0],
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

    x = ttnn.concat(y, 3)

    for i in range(len(y)):
        ttnn.deallocate(y[i])

    x, out_h, out_w = Conv(
        device, x, parameters, f"{path}.cv2", c_ * 4, c2, 1, 1, change_shard=True, inp_h=out_h, inp_w=out_w
    )
    return x, out_h, out_w


def Detect_cv2(device, x, parameters, path, c1, c2, k, reg_max, bfloat8=True, inp_h=None, inp_w=None):
    x, out_h, out_w = Conv(device, x, parameters, f"{path}.0", c1, c2, k, bfloat8=bfloat8, inp_h=inp_h, inp_w=inp_w)

    x, out_h, out_w = Conv(device, x, parameters, f"{path}.1", c2, c2, k, bfloat8=bfloat8, inp_h=out_h, inp_w=out_w)

    x, out_h, out_w = Conv(
        device,
        x,
        parameters,
        path,
        c2,
        reg_max,
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        inp_h=out_h,
        inp_w=out_w,
        change_shard=True,
        is_fused=False,
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    return x, out_h, out_w


def DFL(device, x, parameters, path, c1=16):
    b, _, a = x.shape

    x = ttnn.reshape(x, (b, 4, c1, a))

    x = ttnn.softmax(x, dim=2)

    x = ttnn.permute(x, (0, 1, 3, 2))

    x = Conv(
        device,
        x,
        parameters,
        path,
        c1,
        1,
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        inp_h=x.shape[1],
        inp_w=x.shape[2],
        is_dfl=True,
        change_shard=True,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return x


def Detect(device, x, parameters, path, nc=80, ch=()):
    nc = nc
    nl = len(ch)
    reg_max = 16
    no = nc + reg_max * 4

    c2, c3 = max((16, ch[0] // 4, reg_max * 4)), max(ch[0], min(nc, 100))

    for i in range(nl):
        inp_h = inp_w = int(math.sqrt(x[i].shape[2]))
        a = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv2.{i}",
            c1=ch[i],
            c2=c2,
            k=3,
            reg_max=4 * reg_max,
            inp_h=inp_h,
            inp_w=inp_w,
        )[0]
        b = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv3.{i}",
            c1=ch[i],
            c2=c3,
            k=3,
            reg_max=nc,
            bfloat8=True,
            inp_h=inp_h,
            inp_w=inp_w,
        )[0]
        x[i] = ttnn.concat((a, b), dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

    shape = x[0].shape

    anchors, strides = parameters["anchors"], parameters["strides"]

    xi = []
    for i in x:
        i = ttnn.reshape(i, (shape[0], -1, no), memory_config=ttnn.L1_MEMORY_CONFIG)
        xi.append(i)

    x_cat = ttnn.concat(xi, 1, memory_config=ttnn.L1_MEMORY_CONFIG)

    x_cat = ttnn.permute(x_cat, (0, 2, 1), memory_config=ttnn.L1_MEMORY_CONFIG)

    box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, x_cat.shape[2]])
    cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, x_cat.shape[2]])

    dfl = DFL(device, box, parameters, f"{path}.dfl")

    dbox = ttnn_decode_bboxes(device, dfl, anchors)
    dbox = dbox * strides

    return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]


def DetectionModel(device, x, parameters, res=(320, 320)):
    x, out_h, out_w = Conv(
        device, x, parameters, "model.0", 3, 48, 3, 2, 1, act_block_h=True, inp_h=res[0], inp_w=res[1]
    )

    x, out_h, out_w = Conv(
        device, x, parameters, "model.1", 48, 96, 3, 2, 1, act_block_h=True, inp_h=out_h, inp_w=out_w
    )

    x, out_h, out_w = C2f(
        device, x, parameters, "model.2", 96, 96, n=2, shortcut=True, act_block_h=True, inp_h=out_h, inp_w=out_w
    )

    x, out_h, out_w = Conv(device, x, parameters, "model.3", 96, 192, 3, 2, inp_h=out_h, inp_w=out_w)

    x, out_h, out_w = C2f(
        device, x, parameters, "model.4", 192, 192, n=4, shortcut=True, bfloat8=True, inp_h=out_h, inp_w=out_w
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    four = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, out_h, out_w = Conv(
        device, x, parameters, "model.5", 192, 384, 3, 2, 1, block_shard=True, inp_h=out_h, inp_w=out_w
    )

    x, out_h, out_w = C2f(
        device, x, parameters, "model.6", 384, 384, n=4, shortcut=True, block_shard=True, inp_h=out_h, inp_w=out_w
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    six = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, out_h, out_w = Conv(
        device, x, parameters, "model.7", 384, 576, 3, 2, 1, block_shard=True, inp_h=out_h, inp_w=out_w
    )

    x, out_h, out_w = C2f(
        device,
        x,
        parameters,
        "model.8",
        576,
        576,
        n=2,
        shortcut=True,
        bfloat8=True,
        inp_h=out_h,
        inp_w=out_w,
        change_shard=True,
    )

    x, out_h, out_w = SPPF(device, x, parameters, "model.9", 576, 576, inp_h=out_h, inp_w=out_w)

    nine = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (1, out_h, out_w, x.shape[-1]))

    x = ttnn.upsample(x, scale_factor=(2, 2))

    inp_h, inp_w = x.shape[1], x.shape[2]

    x = ttnn.reshape(x, (1, 1, inp_h * inp_w, x.shape[-1]))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    x = ttnn.concat([x, six], dim=3)

    ttnn.deallocate(six)

    x, out_h, out_w = C2f(
        device, x, parameters, "model.12", 960, 384, n=2, shortcut=False, bfloat8=True, inp_h=inp_h, inp_w=inp_w
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    twelve = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = ttnn.reshape(x, (1, out_h, out_w, x.shape[-1]))

    x = ttnn.upsample(x, scale_factor=(2, 2))

    inp_h, inp_w = x.shape[1], x.shape[2]

    x = ttnn.reshape(x, (1, 1, inp_h * inp_w, x.shape[-1]))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    x = ttnn.concat([x, four], dim=3)
    ttnn.deallocate(four)

    x, out_h, out_w = C2f(device, x, parameters, "model.15", 576, 192, n=2, shortcut=False, inp_h=inp_h, inp_w=inp_w)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    fifteen = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, out_h, out_w = Conv(device, x, parameters, "model.16", 192, 192, 3, 2, 1, inp_h=out_h, inp_w=out_w)

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    x = ttnn.concat([x, twelve], dim=3)
    ttnn.deallocate(twelve)

    x, out_h, out_w = C2f(
        device, x, parameters, "model.18", 576, 384, n=2, shortcut=False, bfloat8=True, inp_h=out_h, inp_w=out_w
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    eighteen = ttnn.clone(x, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    x, out_h, out_w = Conv(
        device, x, parameters, "model.19", 384, 384, 3, 2, 1, block_shard=True, inp_h=out_h, inp_w=out_w
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

    x = ttnn.concat([x, nine], dim=3)
    ttnn.deallocate(nine)

    x, out_h, out_w = C2f(
        device, x, parameters, "model.21", 960, 576, n=2, shortcut=False, bfloat8=True, inp_h=out_h, inp_w=out_w
    )

    x = [fifteen, eighteen, x]

    x = Detect(device, x, parameters, "model.22", nc=80, ch=(192, 384, 576))

    return x


def YOLOv8m(device, x, parameters):
    return DetectionModel(device, x, parameters)

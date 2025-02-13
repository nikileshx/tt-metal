# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class Conv:
    def __init__(
        self,
        device,
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
        block_shard=False,
        c1=None,
        c2=None,
        reshard=True,
    ):
        self.device = device
        self.conv_weight, self.conv_bias = parameters[path]
        self.kernel_size = (k, k)
        self.output_layout = output_layout
        self.memory_config = memory_config
        self.batch_size = batch_size
        self.change_shard = change_shard
        self.width_shard = width_shard
        self.block_shard = block_shard
        self.reshard = reshard
        self.c1 = c1
        self.c2 = c2

        self.conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            activation="relu",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            input_channels_alignment=16 if c1 < 16 else 32,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=True,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
        )

        if self.width_shard:
            self.conv_config.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        if self.block_shard:
            self.conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        if self.change_shard:
            self.conv_config.shard_layout = None

        self.input_memory_config = (
            ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG if True and not width_shard else ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG
        )

        self.conv_kwargs = {
            "in_channels": c1,
            "out_channels": c2,
            "batch_size": self.batch_size,
            "input_height": inp_h,
            "input_width": inp_w,
            "kernel_size": self.kernel_size,
            "stride": (s, s),
            "padding": (p, p),
            "dilation": (1, 1),
            "groups": 1,
            "device": self.device,
            "conv_config": self.conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.conv_weight):
            conv_weight = ttnn.prepare_conv_weights(
                weight_tensor=self.conv_weight,
                weights_format="OIHW",
                input_memory_config=self.input_memory_config,
                input_layout=ttnn.TILE_LAYOUT,
                has_bias=True,
                **self.conv_kwargs,
            )

            conv_bias = ttnn.prepare_conv_bias(
                bias_tensor=self.conv_bias,
                input_memory_config=self.input_memory_config,
                input_layout=ttnn.TILE_LAYOUT,
                **self.conv_kwargs,
            )
            self.conv_weight = ttnn.to_device(conv_weight, device)
            self.conv_bias = ttnn.to_device(conv_bias, device)

    def __str__(self) -> str:
        return f"Conv: {self.conv_weight.shape} {self.conv_bias.shape} {self.kernel_size}"

    def __call__(self, x):
        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        [x, [out_height, out_width]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            **self.conv_kwargs,
            compute_config=compute_config,
            conv_op_cache={},
            debug=False,
            memory_config=self.memory_config,
            return_weights_and_bias=False,
            return_output_dim=True,
        )

        return x, out_height, out_width


class Linear:
    def __init__(
        self,
        device,
        parameters,
        path,
        activation=None,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=None,
    ):
        self.activation = activation
        self.memory_config = memory_config
        self.program_config = program_config

        self.linear_weight, self.linear_bias = parameters[path]

        if not ttnn.is_tensor_storage_on_device(self.linear_weight):
            self.linear_weight = ttnn.to_device(self.linear_weight, device)
            self.linear_bias = ttnn.to_device(self.linear_bias, device)

    def __call__(self, x):
        x = ttnn.linear(
            x,
            self.linear_weight,
            bias=self.linear_bias,
            activation=self.activation,
            memory_config=self.memory_config,
            program_config=self.program_config,
        )
        return x


class TT_Alexnet:
    def __init__(self, device, input_shape, parameters):
        self.batch_size = input_shape[0]
        self.parameters = parameters
        self.device = device
        self.conv1 = Conv(
            device,
            self.parameters,
            "features.0",
            inp_h=224,
            inp_w=224,
            k=11,
            s=4,
            p=2,
            batch_size=self.batch_size,
            c1=3,
            c2=64,
        )

        self.conv2 = Conv(
            device,
            self.parameters,
            "features.3",
            inp_h=27,
            inp_w=27,
            k=5,
            s=1,
            p=2,
            batch_size=self.batch_size,
            c1=64,
            c2=192,
        )

        self.conv3 = Conv(
            device,
            self.parameters,
            "features.6",
            inp_h=13,
            inp_w=13,
            k=3,
            s=1,
            p=1,
            batch_size=self.batch_size,
            c1=192,
            c2=384,
        )
        self.conv4 = Conv(
            device,
            self.parameters,
            "features.8",
            inp_h=13,
            inp_w=13,
            k=3,
            s=1,
            p=1,
            batch_size=self.batch_size,
            c1=384,
            c2=256,
        )

        self.conv5 = Conv(
            device,
            self.parameters,
            "features.10",
            inp_h=13,
            inp_w=13,
            k=3,
            s=1,
            p=1,
            batch_size=self.batch_size,
            c1=256,
            c2=256,
        )

        self.linear1 = Linear(device, self.parameters, "classifier.1")

        self.linear2 = Linear(device, self.parameters, "classifier.4")

        matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=2,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        self.linear3 = Linear(device, self.parameters, "classifier.6", program_config=matmul_config)

    def __call__(self, x):
        x, out_height, out_width = self.conv1(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=out_height,
            input_w=out_width,
            channels=x.shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )

        x, out_height, out_width = self.conv2(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=out_height,
            input_w=out_width,
            channels=x.shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )

        x, out_height, out_width = self.conv3(x)

        x, out_height, out_width = self.conv4(x)

        x, out_height, out_width = self.conv5(x)

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.batch_size,
            input_h=out_height,
            input_w=out_width,
            channels=x.shape[-1],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )

        x = ttnn.reshape(x, (self.batch_size, 6, 6, 256), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.adaptive_avg_pool2d(x, ttnn.Shape([6, 6]), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (self.batch_size, -1), memory_config=ttnn.L1_MEMORY_CONFIG)

        x = self.linear1(x)
        x = ttnn.relu(x)

        x = self.linear2(x)
        x = ttnn.relu(x)

        x = self.linear3(x)
        return x

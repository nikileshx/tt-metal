// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "upsample.hpp"
#include "device/upsample_op.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::upsample {

ttnn::Tensor ExecuteUpSample::invoke(
    const ttnn::Tensor& input_tensor,
    std::variant<int, tt::tt_metal::Array2D> scale_factor,
    const std::string& mode,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<ttnn::Shape>& compute_shape) {
    MemoryConfig mem_config = output_mem_config.value_or(input_tensor.memory_config());
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    auto computation_shape = input_tensor.get_logical_shape();

    if (compute_shape.has_value()) {
        const auto& shape = compute_shape.value();
        TT_FATAL(shape.rank() == 4, "Invalid Compute shape, expected rank 4 but got rank {}", shape.rank());
        computation_shape = shape;

        TT_FATAL(
            computation_shape.volume() == input_tensor.get_logical_shape().volume(),
            "Input and output volumes must match");
    }

    int scale_h = 1;
    int scale_w = 1;
    std::visit(
        [&scale_h, &scale_w](auto&& sf) {
            using T = std::decay_t<decltype(sf)>;
            if constexpr (std::is_same_v<T, int>) {
                scale_h = sf;
                scale_w = sf;
            } else if constexpr (std::is_same_v<T, tt::tt_metal::Array2D>) {
                scale_h = sf.at(0);
                scale_w = sf.at(1);
            } else {
                // static_assert(false, "Unsupported scale factor");
                static_assert(sizeof(T) != 0, "Type check failed.");
            }
        },
        scale_factor);

    // DEBUG
    // fmt::print("scale_h: {}, scale_w: {}\n", scale_h, scale_w);

    if (input_tensor.is_sharded()) {
        // TT_FATAL(not input_tensor.is_sharded(), "Error");
        int shard_height = input_tensor.memory_config().shard_spec.value().shape[0];
        const auto& input_shape = input_tensor.get_logical_shape();
        const auto batch_size = input_shape[0];
        const auto input_h = input_shape[1];
        const auto input_w = input_shape[2];
        const auto num_channels = input_shape[3];
        if (shard_height % input_w != 0) {
            TT_FATAL(shard_height % input_w != 0, "Error");
        }
    }

    return tt::tt_metal::operation::run(
               UpSample{scale_h, scale_w, computation_shape, mode, mem_config, config}, {input_tensor})
        .front();
}

}  // namespace ttnn::operations::upsample

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama.hpp"

#include "device/rotary_embedding_llama_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

Tensor RotaryEmbeddingLlamaOperation::invoke(
    const Tensor& input_tensor,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    const bool is_decode_mode,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    std::vector<Tensor> output_tensors = {
        Tensor(tt::tt_metal::operation::get_workers_for_op_output({input_tensor, cos_cache, sin_cache, trans_mat}))};
    tt::tt_metal::operation::launch_op(
        [is_decode_mode, memory_config, compute_kernel_config](
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<std::optional<Tensor>>& optional_output_tensors) mutable -> std::vector<Tensor> {
            auto& input_tensor = input_tensors.at(0);

            auto arch = input_tensor.storage_type() == StorageType::DEVICE
                            ? input_tensor.device()->arch()
                            : ttnn::operations::experimental::auto_format::AutoFormat::GetDefaultDevice()->arch();
            auto kernel_config_val =
                init_device_compute_kernel_config(arch, compute_kernel_config, MathFidelity::HiFi4, true, false, false);

            tt::tt_metal::MemoryConfig default_memory_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
            if (input_tensor.storage_type() == StorageType::DEVICE) {
                default_memory_config = input_tensor.memory_config();
            }

            return tt::tt_metal::operation::run(
                tt::tt_metal::RotaryEmbeddingLlama{
                    is_decode_mode, memory_config.value_or(default_memory_config), kernel_config_val},
                input_tensors);
        },
        {input_tensor, cos_cache, sin_cache, trans_mat},
        output_tensors);
    return output_tensors.at(0);
}

}  // namespace ttnn::operations::experimental::transformer

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/common/constants.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

namespace ttnn::operations::experimental::transformer {

operation::ProgramWithCallbacks multi_core_nlp_kv_cache_load_slice(
    const Tensor& a,
    Tensor& output,
    const ttnn::SimpleShape& output_tensor_start,
    const ttnn::SimpleShape& output_tensor_end);

struct NlpKVCacheLoadSliceDeviceOperation {
    const ttnn::SimpleShape output_tensor_start;
    const ttnn::SimpleShape output_tensor_end;
    const ttnn::SimpleShape output_shape;
    const ttnn::SimpleShape input_shape;

    void validate(const std::vector<Tensor>& input_tensors) const;
    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;
    operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::transformer

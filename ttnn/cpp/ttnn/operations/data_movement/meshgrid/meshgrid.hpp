// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::data_movement {

struct MeshgridOperation {
    static std::vector<ttnn::Tensor> invoke(const std::vector<ttnn::Tensor>& input_tensors, int64_t indexing);
};

}  // namespace operations::data_movement

constexpr auto meshgrid =
    ttnn::register_operation<"ttnn::meshgrid", ttnn::operations::data_movement::MeshgridOperation>();

}  // namespace ttnn

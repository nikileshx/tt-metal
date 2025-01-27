// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "meshgrid.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> MeshgridOperation::invoke(const std::vector<ttnn::Tensor>& input_tensors, int64_t indexing) {
    std::cout << "Inside the implemented Op\n";
    std::vector<ttnn::Tensor> output_tensors;
    return output_tensors;
}

}  // namespace ttnn::operations::data_movement

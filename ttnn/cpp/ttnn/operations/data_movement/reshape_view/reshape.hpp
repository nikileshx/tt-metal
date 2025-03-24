// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

struct ReshapeViewOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::Shape& shape, const std::optional<ttnn::Layout> layout = std::nullopt);
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, const ttnn::SimpleShape& logical_shape, const std::optional<ttnn::Layout> layout = std::nullopt );
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor, tt::stl::Span<const int32_t> shape_vector, const std::optional<ttnn::Layout> layout = std::nullopt);
};


}  // namespace operations::data_movement

constexpr auto reshape = ttnn::register_operation<"ttnn::reshape", ttnn::operations::data_movement::ReshapeViewOperation>();

}  // namespace ttnn

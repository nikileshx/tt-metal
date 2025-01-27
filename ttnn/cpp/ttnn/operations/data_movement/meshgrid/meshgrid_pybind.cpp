// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "meshgrid_pybind.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cpp/pybind11/decorators.hpp"
#include "ttnn/operations/data_movement/meshgrid/meshgrid.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement {

namespace detail {

template <typename data_movement_operation_t>
void bind_meshgrid(pybind11::module& module, const data_movement_operation_t& operation, const char* doc) {
    bind_registered_operation(
        module,
        operation,
        doc,
        ttnn::pybind_overload_t{
            [](const data_movement_operation_t& self,
               const std::vector<ttnn::Tensor>& input_tensors,
               const std::string& indexing) -> std::vector<ttnn::Tensor> {
                // Convert indexing str to int
                int64_t indexing_int = 0;  // Default to ij
                if (indexing == "xy") {
                    indexing_int = 1;
                } else if (indexing != "ij") {
                    throw std::invalid_argument("Invalid index mode. Must be 'ij' or 'xy'.");
                }
                return self(input_tensors, indexing_int);  // Pass indexing_int directly
            },
            py::arg("input_tensors"),
            py::arg("indexing") = "ij"});
}

}  // namespace detail

void py_bind_meshgrid(pybind11::module& module) {
    detail::bind_meshgrid(
        module,
        ttnn::meshgrid,
        R"doc(meshgrid(input_tensors: List[ttnn.Tensor], indexing: str = 'ij') -> List[ttnn.Tensor]

      Creates grids of coordinates specified by the 1D inputs in input_tensors.

      Args:
          * :attr:`input_tensors`: A list of 1D input tensors.
          * :attr:`indexing`: The indexing mode, either "ij" (default) or "xy".
              - "ij": Dimensions are in the same order as the input tensors.
              - "xy": The first dimension corresponds to the cardinality of the second input,
                  and the second dimension corresponds to the cardinality of the first input (Cartesian coordinate system).

      Returns:
          * :attr:`List[ttnn.Tensor]`: A list of tensors representing the grid of coordinates, one tensor per input dimension.
              Each tensor has a shape matching the lengths of all input tensors.

      Note:
          - If the input tensors are of sizes S0, S1, ..., SN-1, the resulting tensors will each have the shape (S0, S1, ..., SN-1).

      Examples:
          >>> x = torch.tensor([1, 2, 3])
          >>> y = torch.tensor([4, 5, 6])
          >>> grid_x, grid_y = ttnn.meshgrid([x, y], indexing='ij')
          >>> grid_x
          >>> tensor([[1, 1, 1],
                     [2, 2, 2],
                     [3, 3, 3]])
          >>> grid_y
          >>> tensor([[4, 5, 6],
                     [4, 5, 6],
                     [4, 5, 6]])

      )doc");
}

}  // namespace ttnn::operations::data_movement

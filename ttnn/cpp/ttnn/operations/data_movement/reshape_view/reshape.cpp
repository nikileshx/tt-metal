// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0


#include "ttnn/common/constants.hpp"
#include "ttnn/run_operation.hpp"
#include "reshape.hpp"
#include "tt_metal/common/constants.hpp"
#include <functional>
#include <ttnn/operations/numpy/functions.hpp>
#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/data_transfer/data_transfer.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement {


namespace detail {

ttnn::Tensor host_reshape(const ttnn::Tensor& tensor, const ttnn::Shape& shape, const ttnn::Layout output_layout) {
    if (!ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) {
        return tensor.reshape(shape);
    }
    auto tensor_shape = tensor.shape();
    auto layout = tensor.layout();
    auto device = tensor.device();
    auto memory_config = tensor.memory_config();
    auto host_tensor = tensor.cpu();
    auto rm_tensor = ttnn::to_layout(host_tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);

    if(tensor_shape.has_tile_padding()) {

        ttnn::Tensor slice_input;
        auto host_tensor_4d = unsqueeze_to_4D(rm_tensor);
        auto tensor_shape_4d = host_tensor_4d.shape();
        ttnn::SmallVector<uint32_t> begins({0, 0, 0, 0});
        ttnn::SmallVector<uint32_t> ends({tensor_shape_4d[0], tensor_shape_4d[1], tensor_shape_4d[2], tensor_shape_4d[3]});
        ttnn::SmallVector<uint32_t> step({1, 1, 1, 1});
        host_tensor_4d = ttnn::slice(host_tensor_4d, begins, ends, step, std::nullopt);
        host_tensor = squeeze_from_4D(host_tensor_4d, tensor_shape.rank());
    }

    std::cout << " Host Reshape " <<std::endl;
    auto host_reshape_tensor = rm_tensor.reshape(shape);
    auto final_layout_tensor = ttnn::to_layout(host_reshape_tensor, output_layout, std::nullopt, std::nullopt, (Device *)nullptr);
    auto device_tensor = ttnn::data_transfer_to_device(final_layout_tensor, device, memory_config);
    return device_tensor;
}

ttnn::Tensor convert_tensor_to_rm_reshape_convert_back_to_orig_layout(const ttnn::Tensor& tensor, const ttnn::Shape& shape, const ttnn::Layout output_layout) {
    const auto layout = tensor.get_layout();
    auto shape_with_padding = shape.padded_shape();
    auto tensor_shape = tensor.get_shape();
    auto tensor_shape_with_padding = tensor_shape.padded_shape();

    std::cout << "Reshape to Row Major" << std::endl;
    //Constraint in device kernel

    uint32_t ROW_MAJOR_WIDTH = 32/tensor.element_size();
    ttnn::Tensor reshaped_rm_tensor;
    if(tensor.element_size()<2 && layout == ttnn::TILE_LAYOUT)
    {
        //Can't call to_layout on 4b and 8b datatypes
        reshaped_rm_tensor = host_reshape(tensor, shape, output_layout);
    }
    else if((tensor_shape[-1] % ROW_MAJOR_WIDTH == 0 && shape[-1] % ROW_MAJOR_WIDTH == 0)) {
        auto rm_tensor = ttnn::to_layout(tensor, ttnn::ROW_MAJOR_LAYOUT, std::nullopt, std::nullopt, (Device *)nullptr);
        if (rm_tensor.is_contiguous()) {
            // Page size depends on the width, so only modify the shape if the width is the same
            if (tensor_shape_with_padding[-1] == shape_with_padding[-1]) {
                reshaped_rm_tensor =  rm_tensor.reshape(shape);
            }
            //Different page width, going to use device kernel that does transpose
            else {
                auto original_rank = shape.rank();
                auto tensor_4d = unsqueeze_to_4D(rm_tensor);
                const auto shape_4d = shape.to_rank(4);
                auto reshaped_tensor = ttnn::reshape_on_device(tensor_4d, ttnn::SimpleShape{shape_4d[0], shape_4d[1], shape_4d[2], shape_4d[3]}, tensor.memory_config());
                reshaped_rm_tensor = squeeze_from_4D(reshaped_tensor, original_rank);
            }
        } else if (tensor_shape.rank() >= 2 and shape.rank() >= 2) {
            // Handle the case when the tensor is not contiguous but the last two dimensions are the same and so reshape
            // is possible
            if (tensor_shape[-1] == shape[-1] and tensor_shape[-2] == shape[-2] and
                tensor_shape_with_padding[-1] == shape_with_padding[-1] and
                tensor_shape_with_padding[-2] == shape_with_padding[-2]) {
                reshaped_rm_tensor = rm_tensor.reshape(shape);
            }
        } else {
            reshaped_rm_tensor = host_reshape(tensor, shape, output_layout);
        }

    }
    // Can'd do untilize on device due to inner dim size
    else {
        reshaped_rm_tensor = host_reshape(tensor, shape, output_layout);
    }

    if (((shape[-1] * tensor.element_size()) % sizeof(uint32_t) == 0) and reshaped_rm_tensor.layout() != output_layout) {
        return ttnn::to_layout(reshaped_rm_tensor, output_layout, std::nullopt, std::nullopt, (Device *)nullptr);
    }
    else {
        return reshaped_rm_tensor;
    }
}

}

ttnn::Shape tiling_reshape_corrector(const ttnn::Shape& shape) {
    //Apply the correct padding metadata to the target shape
    auto padded = shape.with_tile_padding();
    auto rank = shape.rank();
    const int8_t correction_1 =(ttnn::types::TILE_SIZE - (int)padded[-1] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
    if(rank == 1)
    {
        return ttnn::Shape({1, shape[0]}, {32, padded[0]+correction_1});
    }
    const int8_t correction_2 =(ttnn::types::TILE_SIZE - (int)padded[-2] % ttnn::types::TILE_SIZE) % ttnn::types::TILE_SIZE;
    switch(rank)
    {
        case 2:
            return ttnn::Shape({shape[0],shape[1]},{padded[0]+correction_2,padded[1]+correction_1});
            break;
        case 3:
            return ttnn::Shape({shape[0],shape[1],shape[2]},{padded[0],padded[1]+correction_2,padded[2]+correction_1});
            break;
        case 4:
            return ttnn::Shape({shape[0],shape[1],shape[2],shape[3]},{padded[0],padded[1],padded[2]+correction_2,padded[3]+correction_1});
            break;

    }
    return shape;
}

ttnn::Tensor PerformView(const ttnn::Tensor& tensor, const ttnn::Shape& shape, const ttnn::Layout layout) {
    if (tensor.get_layout() == ttnn::TILE_LAYOUT &&(shape[-1]%ttnn::types::TILE_SIZE!=0 || shape[-2]%ttnn::types::TILE_SIZE!=0 ))
    {
        //Correct the output shape to add padding metadata before reshape (view)
        return tensor.reshape(tiling_reshape_corrector(shape));
    }
    //Perform a reshape (view)
    return tensor.reshape(shape);
}

void Validate_transform (const ttnn::Shape& input_shape, const ttnn::Shape& output_shape)
{
    //Reshape should not be adding or removing data
    uint32_t input_volume = 1;;
    uint32_t output_volume = 1;
    for (int i=0; i <input_shape.rank(); i++)
    {
        input_volume = input_volume * input_shape[i];
    }
    for (int i=0; i <output_shape.rank(); i++)
    {
        output_volume = output_volume * output_shape[i];
    }
    TT_FATAL(input_volume == output_volume, "Invalid Reshape, input and output volume must match");
}

ttnn::Tensor ReshapeViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::Shape& shape, const std::optional<ttnn::Layout> output_layout) {

    auto layout = tensor.get_layout();
    auto tensor_shape = tensor.get_shape();

    std::cout << "Reshape on view " << std::endl;

    auto final_layout = output_layout.value_or(layout);

    // First Case, No reshape Required
    if (tensor_shape == shape) {
        return tensor;
    }
    //This is a constraint Torch places on reshape I was assuming, but it causes half of the codebase to fail if added
    //Validate_transform(tensor_shape, shape)
    //For view the following cases work:
    //RM: The last dimension is the same
    //Tiled: The last two dimensions are the same or there is no padding on the second last dimension
    const uint32_t shape_second_last_dim = shape.rank() >= 2 ? shape[-2] : 1;
    const uint32_t tensor_shape_second_last_dim = tensor_shape.rank() >= 2 ? tensor_shape[-2] : 1;
    bool this_is_view =  (tensor.get_layout() == final_layout ) && (tensor_shape[-1] == shape[-1]) &&
        ((tensor.get_layout() == ttnn::ROW_MAJOR_LAYOUT) || //Its row major
        (shape_second_last_dim==tensor_shape_second_last_dim) || //Second last dimension is the same
        (shape_second_last_dim%ttnn::types::TILE_SIZE==0 && tensor_shape_second_last_dim%ttnn::types::TILE_SIZE==0)); //There is no padding on the second last dimension

    bool tile_tensor_view_reshape_possible = (tensor.get_layout() == final_layout ) && (layout == ttnn::Layout::TILE and
        shape.with_tile_padding().rank() >= 2 and shape.with_tile_padding()[-2] % ttnn::TILE_SIZE == 0 and shape.with_tile_padding()[-1] % ttnn::TILE_SIZE == 0 and
        tensor_shape.with_tile_padding()[-1] == shape.with_tile_padding()[-1]
        );

    std::cout << "this is view " << this_is_view << std::endl;
    std::cout << "tile_tensor_view_reshape_possible " << tile_tensor_view_reshape_possible << std::endl;
    if (!(ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) or tile_tensor_view_reshape_possible) {
        //This case has been allowed in the past though it means introducing padding values to the data

        std::cout <<  "tile_tensor_view_reshape_possible 0" << std::endl;
        return tensor.reshape(shape);
    }
    if (!(ttnn::has_storage_type_of(tensor, ttnn::StorageType::DEVICE)) or this_is_view) {
        std::cout   <<  " Has this view" << std::endl;
        return PerformView(tensor,shape, output_layout.value_or(layout));
    }
    if (tensor_shape.rank() >3)
    {
        uint32_t mult_factor = 1;
        for (int i=0; i <tensor_shape.rank()-3; i++)
        {
            mult_factor = mult_factor * tensor_shape[i];
        }
        const ttnn::Tensor temp_tensor = PerformView(tensor,ttnn::Shape{tensor_shape[-3]*mult_factor,tensor_shape[-2],tensor_shape[-1]}, output_layout.value_or(layout));
        return detail::convert_tensor_to_rm_reshape_convert_back_to_orig_layout(temp_tensor, shape, output_layout.value_or(layout));
    }
    // Catch-all
    // Do the reshape in row-major

    return detail::convert_tensor_to_rm_reshape_convert_back_to_orig_layout(tensor, shape, output_layout.value_or(layout));
}

ttnn::Tensor ReshapeViewOperation::invoke(const ttnn::Tensor& tensor, const ttnn::SimpleShape& shape, const std::optional<ttnn::Layout> layout) {
    return invoke(tensor, ttnn::Shape(shape.view()), layout);
}

ttnn::Tensor ReshapeViewOperation::invoke(
    const ttnn::Tensor& tensor,
    tt::stl::Span<const int32_t> shape_vector,
    const std::optional<ttnn::Layout> layout
    ) {
    return invoke(tensor, tt::tt_metal::infer_dims_for_reshape(tensor, shape_vector), layout);
}

} // ttnn::operations::data_movement namespace

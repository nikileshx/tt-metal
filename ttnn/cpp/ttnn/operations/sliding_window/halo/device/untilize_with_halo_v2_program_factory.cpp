// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_halo_v2_program_factory.hpp"

#include <math.h>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

// In order to make circular buffer indicies sequential, we use variable to keep track of the next available index.
// Circular buffer indices should be assigned right before their creation.
struct CBIndices {
    // Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
    // Not assigning get_next_cb_index() value before creating cb will throw exception in circular_buffer_types.cpp
    // which can be used as a reminder.
    uint32_t src_cb_id = 32;
    uint32_t pad_cb_id = 32;
    uint32_t out_cb_id = 32;

    // Additional CBs for sharded data kernel configs
    uint32_t padding_config_cb_id1 = 32;
    uint32_t padding_config_cb_id2 = 32;
    uint32_t local_config_cb_id1 = 32;
    uint32_t local_config_cb_id2 = 32;
    uint32_t remote_config_cb_id1 = 32;
    uint32_t remote_config_cb_id2 = 32;
    uint32_t untilize_out_cb_id = 32;
    uint32_t get_next_cb_id() { return next_cb_id++; }

private:
    uint32_t next_cb_id = tt::CBIndex::c_0;
};

operation::ProgramWithCallbacks untilize_with_halo_multi_core_v2(
    Program& program,
    const Tensor& input_tensor,
    const uint32_t pad_val,
    const uint32_t ncores_nhw,
    const uint32_t max_out_nsticks_per_core,
    const Tensor& padding_config1,
    const Tensor& padding_config2,
    const Tensor& local_config1,
    const Tensor& local_config2,
    const Tensor& remote_config1,
    const Tensor& remote_config2,
    const bool remote_read,
    const bool transpose_mcast,
    Tensor& output_tensor,
    const bool capture_buffers) {
    IDevice* device = input_tensor.device();
    Buffer* src_buffer = input_tensor.buffer();
    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool skip_untilize = input_tensor.get_layout() == Layout::ROW_MAJOR;

    auto input_shape = input_tensor.get_padded_shape();
    auto output_shape = output_tensor.get_padded_shape();

    tt::DataFormat in_df = datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat out_df = datatype_to_dataformat_converter(output_tensor.get_dtype());
    uint32_t out_nbytes = datum_size(out_df);

    CoreRangeSet all_cores = output_tensor.shard_spec().value().grid;
    auto input_shard_shape = output_tensor.shard_spec().value().shape;
    auto output_shard_shape = output_tensor.shard_spec().value().shape;
    TT_ASSERT(input_shard_shape[1] == output_shard_shape[1]);
    uint32_t input_nhw_height = input_shape[0] * input_shape[1] * input_shape[2];
    uint32_t remapped_input_shard_shape_for_output_grid = tt::div_up(input_nhw_height, ncores_nhw);
    uint32_t ntiles_per_block = tt::div_up(input_shard_shape[1], TILE_WIDTH);
    uint32_t input_nblocks_per_core = tt::div_up(remapped_input_shard_shape_for_output_grid, TILE_HEIGHT);
    uint32_t input_npages = ntiles_per_block * input_nblocks_per_core;

    uint32_t out_stick_nbytes = output_shard_shape[1] * out_nbytes;

    uint32_t in_page_size = tt::tt_metal::detail::TileSize(in_df);
    uint32_t out_tile_size = tt::tt_metal::detail::TileSize(out_df);

    if (skip_untilize) {
        uint32_t in_nbytes = datum_size(in_df);
        in_page_size = input_shard_shape[1] * in_nbytes;
        input_npages = remapped_input_shard_shape_for_output_grid;
    }
    // Construct CBs
    // //
    CBIndices cb_indices = CBIndices();
    cb_indices.src_cb_id = cb_indices.get_next_cb_id();
    // input CB (sharded)
    auto src_cb_config = CircularBufferConfig(input_npages * in_page_size, {{cb_indices.src_cb_id, in_df}})
                             .set_page_size(cb_indices.src_cb_id, in_page_size)
                             .set_globally_allocated_address(*src_buffer);
    auto src_cb = CreateCircularBuffer(program, all_cores, src_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.src_cb_id, input_npages, in_page_size);

    uint32_t input_to_writer_cb_id = cb_indices.src_cb_id;
    if (!skip_untilize) {
        cb_indices.untilize_out_cb_id = cb_indices.get_next_cb_id();
        input_to_writer_cb_id = cb_indices.untilize_out_cb_id;
        // output of untilize from compute kernel goes into this CB
        uint32_t output_ntiles = ntiles_per_block * input_nblocks_per_core;
        auto untilize_out_cb_config =
            CircularBufferConfig(output_ntiles * out_tile_size, {{cb_indices.untilize_out_cb_id, out_df}})
                .set_page_size(cb_indices.untilize_out_cb_id, out_tile_size);
        auto untilize_out_cb = CreateCircularBuffer(program, all_cores, untilize_out_cb_config);
        log_debug(
            tt::LogOp,
            "CB {} :: npages = {}, pagesize = {}",
            cb_indices.untilize_out_cb_id,
            output_ntiles,
            out_tile_size);
    }

    cb_indices.out_cb_id = cb_indices.get_next_cb_id();
    // output shard, after inserting halo and padding, goes into this CB as input to next op.
    uint32_t out_cb_pagesize = out_stick_nbytes;
    uint32_t out_cb_npages = max_out_nsticks_per_core;
    auto out_cb_config = CircularBufferConfig(out_cb_npages * out_cb_pagesize, {{cb_indices.out_cb_id, out_df}})
                             .set_page_size(cb_indices.out_cb_id, out_cb_pagesize)
                             .set_globally_allocated_address(*dst_buffer);
    auto out_cb = CreateCircularBuffer(program, all_cores, out_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.out_cb_id, out_cb_npages, out_cb_pagesize);

    // CB for pad val buffer (stick sized)
    uint32_t pad_cb_pagesize = out_stick_nbytes;
    uint32_t pad_cb_npages = 1;
    cb_indices.pad_cb_id = cb_indices.get_next_cb_id();
    auto pad_cb_config = CircularBufferConfig(pad_cb_pagesize * pad_cb_npages, {{cb_indices.pad_cb_id, out_df}})
                             .set_page_size(cb_indices.pad_cb_id, pad_cb_pagesize);
    auto pad_cb = CreateCircularBuffer(program, all_cores, pad_cb_config);
    log_debug(tt::LogOp, "CB {} :: npages = {}, pagesize = {}", cb_indices.pad_cb_id, pad_cb_npages, pad_cb_pagesize);

    tt::DataFormat kernel_config_df = tt::DataFormat::RawUInt16;  // NOTE: UInt16 is not supported for CB types
    uint32_t config_nbytes =
        tt::datum_size(kernel_config_df) * 2;  // each config is a pair "start, size", so double the size
    uint32_t pagesize = 0;

    // Gather data
    if (!skip_untilize) {
        // compute kernel
        std::vector<uint32_t> compute_ct_args = {
            input_nblocks_per_core, ntiles_per_block, cb_indices.src_cb_id, input_to_writer_cb_id};
        std::string compute_kernel(
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
        if (ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH) {
            log_debug(
                tt::LogOp,
                "Falling back to slow untilize since ntiles_per_block {} > MAX_PACK_UNTILIZE_WIDTH {}",
                ntiles_per_block,
                MAX_PACK_UNTILIZE_WIDTH);
            compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        }
        KernelHandle untilize_kernel_id =
            CreateKernel(program, compute_kernel, all_cores, ComputeConfig{.compile_args = compute_ct_args});
    }

    TT_ASSERT(padding_config1.get_dtype() == DataType::UINT16);
    TT_ASSERT(padding_config2.get_dtype() == DataType::UINT16);
    TT_ASSERT(local_config1.get_dtype() == DataType::UINT16);
    TT_ASSERT(local_config2.get_dtype() == DataType::UINT16);
    TT_ASSERT(remote_config1.get_dtype() == DataType::UINT16);
    TT_ASSERT(remote_config2.get_dtype() == DataType::UINT16);

    auto padding_config_buffer1 = padding_config1.device_buffer();
    const uint32_t num_cores = all_cores.num_cores();
    cb_indices.padding_config_cb_id1 = cb_indices.get_next_cb_id();
    auto padding_config_cb_config1 =
        CircularBufferConfig(
            padding_config_buffer1->size() / num_cores, {{cb_indices.padding_config_cb_id1, kernel_config_df}})
            .set_page_size(cb_indices.padding_config_cb_id1, padding_config_buffer1->page_size())
            .set_globally_allocated_address(*padding_config_buffer1);
    CBHandle padding_config_cb1 = CreateCircularBuffer(program, all_cores, padding_config_cb_config1);

    cb_indices.padding_config_cb_id2 = cb_indices.get_next_cb_id();
    auto padding_config_buffer2 = padding_config2.device_buffer();
    auto padding_config_cb_config2 =
        CircularBufferConfig(
            padding_config_buffer2->size() / num_cores, {{cb_indices.padding_config_cb_id2, kernel_config_df}})
            .set_page_size(cb_indices.padding_config_cb_id2, padding_config_buffer2->page_size())
            .set_globally_allocated_address(*padding_config_buffer2);
    CBHandle padding_config_cb2 = CreateCircularBuffer(program, all_cores, padding_config_cb_config2);

    cb_indices.local_config_cb_id1 = cb_indices.get_next_cb_id();
    auto local_config_buffer1 = local_config1.device_buffer();
    auto local_config_cb_config1 =
        CircularBufferConfig(
            local_config_buffer1->size() / num_cores, {{cb_indices.local_config_cb_id1, kernel_config_df}})
            .set_page_size(cb_indices.local_config_cb_id1, local_config_buffer1->page_size())
            .set_globally_allocated_address(*local_config_buffer1);
    CBHandle local_config_cb1 = CreateCircularBuffer(program, all_cores, local_config_cb_config1);

    cb_indices.local_config_cb_id2 = cb_indices.get_next_cb_id();
    auto local_config_buffer2 = local_config2.device_buffer();
    auto local_config_cb_config2 =
        CircularBufferConfig(
            local_config_buffer2->size() / num_cores, {{cb_indices.local_config_cb_id2, kernel_config_df}})
            .set_page_size(cb_indices.local_config_cb_id2, local_config_buffer2->page_size())
            .set_globally_allocated_address(*local_config_buffer2);
    CBHandle local_config_cb2 = CreateCircularBuffer(program, all_cores, local_config_cb_config2);

    cb_indices.remote_config_cb_id1 = cb_indices.get_next_cb_id();
    auto remote_config_buffer1 = remote_config1.device_buffer();
    auto remote_config_cb_config1 =
        CircularBufferConfig(
            remote_config_buffer1->size() / num_cores, {{cb_indices.remote_config_cb_id1, kernel_config_df}})
            .set_page_size(cb_indices.remote_config_cb_id1, remote_config_buffer1->page_size())
            .set_globally_allocated_address(*remote_config_buffer1);
    CBHandle remote_config_cb1 = CreateCircularBuffer(program, all_cores, remote_config_cb_config1);

    cb_indices.remote_config_cb_id2 = cb_indices.get_next_cb_id();
    auto remote_config_buffer2 = remote_config2.device_buffer();
    auto remote_config_cb_config2 =
        CircularBufferConfig(
            remote_config_buffer2->size() / num_cores, {{cb_indices.remote_config_cb_id2, kernel_config_df}})
            .set_page_size(cb_indices.remote_config_cb_id2, remote_config_buffer2->page_size())
            .set_globally_allocated_address(*remote_config_buffer2);
    CBHandle remote_config_cb2 = CreateCircularBuffer(program, all_cores, remote_config_cb_config2);

    const bool is_block_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::BLOCK_SHARDED;
    const bool is_width_sharded = input_tensor.memory_config().memory_layout == TensorMemoryLayout::WIDTH_SHARDED;

    auto aligned_input_nstick_nbytes = out_stick_nbytes;
    log_debug(tt::LogOp, "out_stick_nbytes = {}", out_stick_nbytes);
    log_debug(tt::LogOp, "input_tensor.buffer()->alignment() = {}", input_tensor.buffer()->alignment());

    if (out_stick_nbytes % input_tensor.buffer()->alignment() != 0) {
        aligned_input_nstick_nbytes = tt::round_up(out_stick_nbytes, input_tensor.buffer()->alignment());
    }
    // reader kernel
    std::vector<uint32_t> reader_ct_args = {
        0,  // padding_config_cb_id
        0,  // local_config_cb_id
        0,  // remote_config_cb_id
        cb_indices.src_cb_id,
        input_to_writer_cb_id,
        cb_indices.out_cb_id,
        cb_indices.pad_cb_id,
        pad_val,
        input_npages,
        out_stick_nbytes,
        is_block_sharded,
        remote_read,
        (uint32_t)(transpose_mcast ? 1 : 0),
        is_width_sharded,
        aligned_input_nstick_nbytes,
        true};

    reader_ct_args[0] = cb_indices.padding_config_cb_id1;
    reader_ct_args[1] = cb_indices.local_config_cb_id2;
    reader_ct_args[2] = cb_indices.remote_config_cb_id1;
    KernelHandle reader_kernel_id0 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = reader_ct_args});

    reader_ct_args[0] = cb_indices.padding_config_cb_id2;
    // Change order of cbs so that in case if total(local_config1 and local_config2)local writes and
    // total(remote_config1 and remote_config2) remote writes both are odd, load is better balanced.
    reader_ct_args[1] = cb_indices.local_config_cb_id1;
    reader_ct_args[2] = cb_indices.remote_config_cb_id2;
    reader_ct_args[15] = false;

    KernelHandle reader_kernel_id1 = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/sliding_window/halo/device/kernels/dataflow/halo_gather.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    if (!capture_buffers) {
        padding_config_buffer1 = nullptr;
        padding_config_buffer2 = nullptr;
        local_config_buffer1 = nullptr;
        local_config_buffer2 = nullptr;
        remote_config_buffer1 = nullptr;
        remote_config_buffer2 = nullptr;
    }
    // Capture padding_config_buffer, local_config_buffer, remote_config_buffer to cache this with the program
    auto override_runtime_arguments_callback = [src_cb,
                                                out_cb,
                                                padding_config_cb1,
                                                padding_config_cb2,
                                                local_config_cb1,
                                                local_config_cb2,
                                                remote_config_cb1,
                                                remote_config_cb2,
                                                padding_config_buffer1,
                                                padding_config_buffer2,
                                                local_config_buffer1,
                                                local_config_buffer2,
                                                remote_config_buffer1,
                                                remote_config_buffer2](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&,
                                                   const std::vector<Tensor>& output_tensors) {
        auto src_buffer = input_tensors.at(0).buffer();
        auto dst_buffer = output_tensors.at(0).buffer();

        UpdateDynamicCircularBufferAddress(program, src_cb, *src_buffer);
        UpdateDynamicCircularBufferAddress(program, out_cb, *dst_buffer);
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <string>
#include <vector>
#include <optional>
#include <lightmetal_binary.hpp>

#include <tt-metalium/device.hpp>

// Forward decl for trace_buffer.hpp
namespace tt::tt_metal {
class TraceDescriptor;
}

// Forward decl for command_generated.h / light_metal_binary_generated.h
namespace tt::tt_metal::flatbuffer {
struct Command;
struct ReplayTraceCommand;
struct EnqueueTraceCommand;
struct LoadTraceCommand;
struct ReleaseTraceCommand;
struct CreateBufferCommand;
struct DeallocateBufferCommand;
struct EnqueueWriteBufferCommand;
struct EnqueueReadBufferCommand;
struct FinishCommand;
struct CreateProgramCommand;
struct EnqueueProgramCommand;
struct CreateKernelCommand;
struct SetRuntimeArgsUint32Command;
struct SetRuntimeArgsCommand;
struct CreateCircularBufferCommand;
struct LightMetalCompareCommand;
struct RuntimeArg;

struct TraceDescriptor;
struct TraceDescriptorByTraceId;
struct LightMetalBinary;
}  // namespace tt::tt_metal::flatbuffer

using FlatbufferRuntimeArgVector =
    const flatbuffers::Vector<flatbuffers::Offset<tt::tt_metal::flatbuffer::RuntimeArg>>*;
using RuntimeArgs = std::vector<std::variant<Buffer*, uint32_t>>;

namespace tt::tt_metal {
inline namespace v0 {

class LightMetalReplay {
public:
    // Constructor that initializes the class with a binary blob and transfers ownership of the blob.
    explicit LightMetalReplay(LightMetalBinary&& binary);

    // Execute the stored LightMetal binary by looping over all commands, and execting them.
    // Returns true if passed.  Currently has no side-effects/artifacts returned to user,
    // may change in the future.
    bool execute_binary();

private:
    // Executor functions for all traced host API calls (commands)
    void execute(const tt::tt_metal::flatbuffer::Command* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReplayTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::LoadTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::ReleaseTraceCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::DeallocateBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueWriteBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueReadBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::FinishCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateProgramCommand* command);
    void execute(const tt::tt_metal::flatbuffer::EnqueueProgramCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateKernelCommand* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsUint32Command* command);
    void execute(const tt::tt_metal::flatbuffer::SetRuntimeArgsCommand* command);
    void execute(const tt::tt_metal::flatbuffer::CreateCircularBufferCommand* command);
    void execute(const tt::tt_metal::flatbuffer::LightMetalCompareCommand* command);

    // Object maps public accessors
    void add_buffer_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Buffer>& buffer);
    std::shared_ptr<::tt::tt_metal::Buffer> get_buffer_from_map(uint32_t global_id) const;
    void remove_bufer_from_map(uint32_t global_id);

    void add_program_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Program>& program);
    std::shared_ptr<::tt::tt_metal::Program> get_program_from_map(uint32_t global_id) const;
    void remove_program_from_map(uint32_t global_id);

    void add_kernel_handle_to_map(uint32_t global_id, ::tt::tt_metal::KernelHandle kernel_id);
    ::tt::tt_metal::KernelHandle get_kernel_handle_from_map(uint32_t global_id) const;
    void remove_kernel_handle_from_map(uint32_t global_id);

    void add_kernel_to_map(uint32_t global_id, const std::shared_ptr<::tt::tt_metal::Kernel>& kernel);
    std::shared_ptr<::tt::tt_metal::Kernel> get_kernel_from_map(uint32_t global_id) const;
    void remove_kernel_from_map(uint32_t global_id);

    void add_cb_handle_to_map(uint32_t global_id, ::tt::tt_metal::CBHandle cb_handle);
    ::tt::tt_metal::CBHandle get_cb_handle_from_map(uint32_t global_id) const;
    void remove_cb_handle_from_map(uint32_t global_id);

    // Return the TraceDescriptor for a given trace_id from flatbuffer.
    std::optional<TraceDescriptor> get_trace_by_id(uint32_t target_trace_id);

    // fromFlatBuffer that need class state
    std::shared_ptr<RuntimeArgs> rt_args_from_flatbuffer(const FlatbufferRuntimeArgVector flatbuffer_args);

    // Workload related members --------------------
    const tt::tt_metal::flatbuffer::LightMetalBinary* parse_flatbuffer_binary();

    LightMetalBinary binary_;                                      // Stored binary blob
    const tt::tt_metal::flatbuffer::LightMetalBinary* fb_binary_;  // Parsed FlatBuffer binary
    bool show_reads_ = false;                                      // Flag to show read buffer contents
    bool disable_checking_ = false;  // Optionally disable equality checking in Compare command.

    // System related members ----------------------
    void setup_devices();
    void close_devices();

    tt::tt_metal::IDevice* device_ = nullptr;

    // Object maps for storing objects by global_id
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Buffer>> buffer_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Program>> program_map_;
    std::unordered_map<uint32_t, tt::tt_metal::KernelHandle> kernel_handle_map_;
    std::unordered_map<uint32_t, std::shared_ptr<::tt::tt_metal::Kernel>> kernel_map_;
    std::unordered_map<uint32_t, tt::tt_metal::CBHandle> cb_handle_map_;
};

}  // namespace v0
}  // namespace tt::tt_metal

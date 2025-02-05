// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <fstream>
#include <memory>
#include <span>
#include <thread>
#include <utility>
#include <vector>

#include "env_lib.hpp"
#include "command_queue_interface.hpp"
#include <tt-metalium/dispatch_settings.hpp>
#include "device_command.hpp"
#include "multi_producer_single_consumer_queue.hpp"
#include "program_command_sequence.hpp"
#include "worker_config_buffer.hpp"
#include "program_impl.hpp"
#include "trace_buffer.hpp"

namespace tt::tt_metal {
inline namespace v0 {

class BufferRegion;
class Event;
class Trace;

}  // namespace v0

// Only contains the types of commands which are enqueued onto the device
enum class EnqueueCommandType {
    ENQUEUE_READ_BUFFER,
    ENQUEUE_WRITE_BUFFER,
    GET_BUF_ADDR,
    ADD_BUFFER_TO_PROGRAM,
    SET_RUNTIME_ARGS,
    ENQUEUE_PROGRAM,
    ENQUEUE_TRACE,
    ENQUEUE_RECORD_EVENT,
    ENQUEUE_WAIT_FOR_EVENT,
    FINISH,
    FLUSH,
    TERMINATE,
    INVALID
};

class Command {
public:
    Command() {}
    virtual void process() {};
    virtual EnqueueCommandType type() = 0;
};

class EnqueueProgramCommand : public Command {
private:
    uint32_t command_queue_id;
    IDevice* device;
    NOC noc_index;
    Program& program;
    SystemMemoryManager& manager;
    WorkerConfigBufferMgr& config_buffer_mgr;
    CoreCoord dispatch_core;
    CoreType dispatch_core_type;
    uint32_t expected_num_workers_completed;
    uint32_t packed_write_max_unicast_sub_cmds;
    uint32_t dispatch_message_addr;
    uint32_t multicast_cores_launch_message_wptr = 0;
    uint32_t unicast_cores_launch_message_wptr = 0;
    // TODO: There will be multiple ids once programs support spanning multiple sub_devices
    SubDeviceId sub_device_id = SubDeviceId{0};

public:
    EnqueueProgramCommand(
        uint32_t command_queue_id,
        IDevice* device,
        NOC noc_index,
        Program& program,
        CoreCoord& dispatch_core,
        SystemMemoryManager& manager,
        WorkerConfigBufferMgr& config_buffer_mgr,
        uint32_t expected_num_workers_completed,
        uint32_t multicast_cores_launch_message_wptr,
        uint32_t unicast_cores_launch_message_wptr,
        SubDeviceId sub_device_id);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_PROGRAM; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueRecordEventCommand : public Command {
private:
    uint32_t command_queue_id;
    IDevice* device;
    NOC noc_index;
    SystemMemoryManager& manager;
    uint32_t event_id;
    tt::stl::Span<const uint32_t> expected_num_workers_completed;
    tt::stl::Span<const SubDeviceId> sub_device_ids;
    bool clear_count;
    bool write_barrier;

public:
    EnqueueRecordEventCommand(
        uint32_t command_queue_id,
        IDevice* device,
        NOC noc_index,
        SystemMemoryManager& manager,
        uint32_t event_id,
        tt::stl::Span<const uint32_t> expected_num_workers_completed,
        tt::stl::Span<const SubDeviceId> sub_device_ids,
        bool clear_count = false,
        bool write_barrier = true);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_RECORD_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

class EnqueueWaitForEventCommand : public Command {
private:
    uint32_t command_queue_id;
    IDevice* device;
    SystemMemoryManager& manager;
    const Event& sync_event;
    CoreType dispatch_core_type;
    bool clear_count;

public:
    EnqueueWaitForEventCommand(
        uint32_t command_queue_id,
        IDevice* device,
        SystemMemoryManager& manager,
        const Event& sync_event,
        bool clear_count = false);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_WAIT_FOR_EVENT; }

    constexpr bool has_side_effects() { return false; }
};

class EnqueueTraceCommand : public Command {
private:
    uint32_t command_queue_id;
    Buffer& buffer;
    IDevice* device;
    SystemMemoryManager& manager;
    std::shared_ptr<TraceDescriptor>& descriptor;
    std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed;
    bool clear_count;
    NOC noc_index;
    CoreCoord dispatch_core;

public:
    EnqueueTraceCommand(
        uint32_t command_queue_id,
        IDevice* device,
        SystemMemoryManager& manager,
        std::shared_ptr<TraceDescriptor>& descriptor,
        Buffer& buffer,
        std::array<uint32_t, DispatchSettings::DISPATCH_MESSAGE_ENTRIES>& expected_num_workers_completed,
        NOC noc_index,
        CoreCoord dispatch_core);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::ENQUEUE_TRACE; }

    constexpr bool has_side_effects() { return true; }
};

class EnqueueTerminateCommand : public Command {
private:
    uint32_t command_queue_id;
    IDevice* device;
    SystemMemoryManager& manager;

public:
    EnqueueTerminateCommand(uint32_t command_queue_id, IDevice* device, SystemMemoryManager& manager);

    void process();

    EnqueueCommandType type() { return EnqueueCommandType::TERMINATE; }

    constexpr bool has_side_effects() { return false; }
};

}  // namespace tt::tt_metal

std::ostream& operator<<(std::ostream& os, const tt::tt_metal::EnqueueCommandType& type);

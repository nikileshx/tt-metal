
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <tuple>
#include <map>

#include "umd/device/types/arch.h"
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "tt_backend_api_types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/df.hpp"
#include "tt_metal/test_utils/env_vars.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

#include <tt-metalium/persistent_kernel_cache.hpp>

// TODO: ARCH_NAME specific, must remove
#include "eth_l1_address_map.h"

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

class N300TestDevice {
public:
    N300TestDevice() : device_open(false) {
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());

        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (arch_ == tt::ARCH::WORMHOLE_B0 and tt::tt_metal::GetNumAvailableDevices() >= 2 and
            tt::tt_metal::GetNumPCIeDevices() >= 1) {
            std::vector<chip_id_t> ids(num_devices_, 0);
            std::iota(ids.begin(), ids.end(), 0);
            devices_ = tt::tt_metal::detail::CreateDevices(ids);

        } else {
            TT_THROW("This suite can only be run on N300 Wormhole devices");
        }
        device_open = true;
    }
    ~N300TestDevice() {
        if (device_open) {
            TearDown();
        }
    }

    void TearDown() {
        device_open = false;
        for (auto [device_id, device_ptr] : devices_) {
            tt::tt_metal::CloseDevice(device_ptr);
        }
    }

    std::map<chip_id_t, IDevice*> devices_;
    tt::ARCH arch_;
    size_t num_devices_;

private:
    bool device_open;
};

void validation(const std::shared_ptr<tt::tt_metal::Buffer>& worker_buffer) {
    std::vector<uint8_t> golden_vec(worker_buffer->size(), 0);
    std::vector<uint8_t> result_vec(worker_buffer->size(), 0);

    for (int i = 0; i < worker_buffer->size(); ++i) {
        golden_vec[i] = i;
    }
    tt::tt_metal::detail::ReadFromBuffer(worker_buffer, result_vec);

    bool pass = golden_vec == result_vec;
    TT_FATAL(pass, "validation failed");
}

std::vector<Program> build(
    IDevice* device0,
    IDevice* device1,
    CoreCoord eth_sender_core,
    CoreCoord eth_receiver_core,
    CoreCoord worker_core,
    std::size_t num_samples,
    std::size_t sample_page_size,
    std::size_t num_buffer_slots,
    KernelHandle& local_kernel,
    KernelHandle& remote_kernel,
    std::shared_ptr<Buffer>& worker_buffer) {
    Program program0;
    Program program1;

    // worker core coords
    uint32_t worker_noc_x = device1->worker_core_from_logical_core(worker_core).x;
    uint32_t worker_noc_y = device1->worker_core_from_logical_core(worker_core).y;

    uint32_t worker_buffer_addr = worker_buffer->address();

    // eth core ct args
    const std::vector<uint32_t>& eth_sender_ct_args = {num_buffer_slots};
    const std::vector<uint32_t>& eth_receiver_ct_args = {
        num_buffer_slots, worker_noc_x, worker_noc_y, worker_buffer_addr};

    // eth core rt args
    const std::vector<uint32_t>& eth_sender_receiver_rt_args = {
        tt_metal::hal.get_dev_addr(HalProgrammableCoreType::ACTIVE_ETH, HalL1MemAddrType::UNRESERVED),
        static_cast<uint32_t>(num_samples),
        static_cast<uint32_t>(sample_page_size)};

    local_kernel = tt_metal::CreateKernel(
        program0,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
        "ethernet_write_worker_latency_ubench_sender.cpp",
        eth_sender_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_sender_ct_args});
    tt_metal::SetRuntimeArgs(program0, local_kernel, eth_sender_core, eth_sender_receiver_rt_args);

    remote_kernel = tt_metal::CreateKernel(
        program1,
        "tests/tt_metal/tt_metal/test_kernels/dataflow/unit_tests/erisc/"
        "ethernet_write_worker_latency_ubench_receiver.cpp",
        eth_receiver_core,
        tt_metal::EthernetConfig{.noc = tt_metal::NOC::RISCV_0_default, .compile_args = eth_receiver_ct_args});
    tt_metal::SetRuntimeArgs(program1, remote_kernel, eth_receiver_core, eth_sender_receiver_rt_args);

    // Launch
    try {
        tt::tt_metal::detail::CompileProgram(device0, program0);
        tt::tt_metal::detail::CompileProgram(device1, program1);
    } catch (std::exception& e) {
        log_error(tt::LogTest, "Failed compile: {}", e.what());
        throw e;
    }

    std::vector<Program> programs;
    programs.push_back(std::move(program0));
    programs.push_back(std::move(program1));
    return programs;
}

void run(
    IDevice* device0, IDevice* device1, Program& program0, Program& program1, std::shared_ptr<Buffer>& worker_buffer) {
    if (std::getenv("TT_METAL_SLOW_DISPATCH_MODE")) {
        std::thread th2 = std::thread([&] { tt_metal::detail::LaunchProgram(device0, program0); });
        std::thread th1 = std::thread([&] { tt_metal::detail::LaunchProgram(device1, program1); });

        th2.join();
        th1.join();
    } else {
        tt_metal::EnqueueProgram(device0->command_queue(), program0, false);
        tt_metal::EnqueueProgram(device1->command_queue(), program1, false);

        log_info(tt::LogTest, "Calling Finish");
        tt_metal::Finish(device0->command_queue());
        tt_metal::Finish(device1->command_queue());
    }
    tt::tt_metal::detail::DumpDeviceProfileResults(device0);
    tt::tt_metal::detail::DumpDeviceProfileResults(device1);

    validation(worker_buffer);
}

int main(int argc, char** argv) {
    std::size_t arg_idx = 1;
    std::size_t num_samples = std::stoi(argv[arg_idx++]);
    std::size_t sample_page_size = std::stoi(argv[arg_idx++]);
    std::size_t num_buffer_slots = std::stoi(argv[arg_idx++]);

    auto arch = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    if (num_devices < 2) {
        log_info(tt::LogTest, "Need at least 2 devices to run this test");
        return 0;
    }
    if (arch == tt::ARCH::GRAYSKULL) {
        log_info(tt::LogTest, "Test must be run on WH");
        return 0;
    }

    log_info(tt::LogTest, "setting up test fixture");
    N300TestDevice test_fixture;
    log_info(tt::LogTest, "done setting up test fixture");

    const auto& device_0 = test_fixture.devices_.at(0);
    const auto& active_eth_cores = device_0->get_active_ethernet_cores(true);
    auto eth_sender_core_iter = active_eth_cores.begin();
    auto eth_sender_core_iter_end = active_eth_cores.end();
    chip_id_t device_id = std::numeric_limits<chip_id_t>::max();
    tt_xy_pair eth_receiver_core;
    bool initialized = false;
    tt_xy_pair eth_sender_core;
    do {
        TT_ASSERT(eth_sender_core_iter != eth_sender_core_iter_end);
        std::tie(device_id, eth_receiver_core) = device_0->get_connected_ethernet_core(*eth_sender_core_iter);
        eth_sender_core = *eth_sender_core_iter;
        eth_sender_core_iter++;
    } while (device_id != 1);

    log_info(tt::LogTest, "eth_sender_core: {}", eth_sender_core);
    log_info(tt::LogTest, "eth_receiver_core: {}", eth_receiver_core);

    TT_ASSERT(device_id == 1);
    const auto& device_1 = test_fixture.devices_.at(device_id);
    // worker
    auto worker_core = CoreCoord(0, 0);
    // Add more configurations here until proper argc parsing added
    bool success = false;
    success = true;
    log_info(tt::LogTest, "STARTING");
    try {
        log_info(
            tt::LogTest,
            "num_samples: {}, sample_page_size: {}, num_buffer_slots: {}",
            num_samples,
            sample_page_size,
            num_buffer_slots);
        KernelHandle local_kernel;
        KernelHandle remote_kernel;
        try {
            ShardSpecBuffer shard_spec = ShardSpecBuffer(
                CoreRangeSet(std::set<CoreRange>({CoreRange(worker_core)})),
                {1, sample_page_size},
                ShardOrientation::ROW_MAJOR,
                {1, sample_page_size},
                {1, sample_page_size});
            auto worker_buffer = CreateBuffer(tt::tt_metal::ShardedBufferConfig{
                .device = device_1,
                .size = sample_page_size,
                .page_size = sample_page_size,
                .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
                .shard_parameters = shard_spec});

            auto programs = build(
                device_0,
                device_1,
                eth_sender_core,
                eth_receiver_core,
                worker_core,
                num_samples,
                sample_page_size,
                num_buffer_slots,
                local_kernel,
                remote_kernel,
                worker_buffer);
            run(device_0, device_1, programs[0], programs[1], worker_buffer);
        } catch (std::exception& e) {
            log_error(tt::LogTest, "Caught exception: {}", e.what());
            test_fixture.TearDown();
            return -1;
        }
    } catch (std::exception& e) {
        test_fixture.TearDown();
        return -1;
    }

    return success ? 0 : -1;
}

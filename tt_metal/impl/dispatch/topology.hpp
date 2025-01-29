// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <device.hpp>

namespace tt::tt_metal {

// Max number of upstream/downstream dispatch kernels that can be connected to a single dispatch kernel.
#define DISPATCH_MAX_UPSTREAM_KERNELS 4
#define DISPATCH_MAX_DOWNSTREAM_KERNELS 4

struct DispatchKernelNode {
    int id;
    chip_id_t device_id;             // Device that this kernel is located on
    chip_id_t servicing_device_id;   // Remote device that this kernel services, used for kernels on MMIO
    uint8_t cq_id;                   // CQ this kernel implements
    DispatchWorkerType kernel_type;  // Type of dispatch kernel this is
    int upstream_ids[DISPATCH_MAX_UPSTREAM_KERNELS];      // Upstream dispatch kernels
    int downstream_ids[DISPATCH_MAX_DOWNSTREAM_KERNELS];  // Downstream dispatch kernels
    NOC my_noc;                                           // NOC this kernel uses to dispatch kernels
    NOC upstream_noc;                                     // NOC used to communicate upstream
    NOC downstream_noc;                                   // NOC used to communicate downstream
};

// Create FD kernels for all given device ids. Creates all objects, but need to call create_and_compile_cq_program() use
// a created Device to fill out the settings. First version automatically generates the topology based on devices, num
// cqs, and detected board. Second version uses the topology passed in.
void populate_fd_kernels(const std::set<chip_id_t>& device_ids, uint32_t num_hw_cqs);
void populate_fd_kernels(const std::vector<DispatchKernelNode>& nodes);

// Fill out all settings for FD kernels on the given device, and add them to a Program and return it.
std::unique_ptr<tt::tt_metal::Program> create_and_compile_cq_program(tt::tt_metal::IDevice* device);

// Perform additional configuration (writing to specific L1 addresses, etc.) for FD kernels on this device.
void configure_dispatch_cores(tt::tt_metal::IDevice* device);

}  // namespace tt::tt_metal

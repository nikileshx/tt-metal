// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <set>

#include <tt-metalium/dev_msgs.h>
#include <tt-metalium/core_descriptor.hpp>
#include "hostdevcommon/dprint_common.h"
#include "impl/dispatch/dispatch_core_manager.hpp"
#include <device.hpp>

namespace tt::tt_metal {

// Helper function for comparing CoreDescriptors for using in sets.
struct CoreDescriptorComparator {
    bool operator()(const CoreDescriptor& x, const CoreDescriptor& y) const {
        if (x.coord == y.coord) {
            return x.type < y.type;
        } else {
            return x.coord < y.coord;
        }
    }
};
using CoreDescriptorSet = std::set<CoreDescriptor, CoreDescriptorComparator>;

// Helper function to get CoreDescriptors for all debug-relevant cores on device.
static CoreDescriptorSet GetAllCores(tt::tt_metal::IDevice* device) {
    CoreDescriptorSet all_cores;
    // The set of all printable cores is Tensix + Eth cores
    CoreCoord logical_grid_size = device->logical_grid_size();
    for (uint32_t x = 0; x < logical_grid_size.x; x++) {
        for (uint32_t y = 0; y < logical_grid_size.y; y++) {
            all_cores.insert({{x, y}, CoreType::WORKER});
        }
    }
    for (const auto& logical_core : device->get_active_ethernet_cores()) {
        all_cores.insert({logical_core, CoreType::ETH});
    }
    for (const auto& logical_core : device->get_inactive_ethernet_cores()) {
        all_cores.insert({logical_core, CoreType::ETH});
    }

    return all_cores;
}

// Helper function to get CoreDescriptors for all cores that are used for dispatch. Should be a subset of
// GetAllCores().
static CoreDescriptorSet GetDispatchCores(tt::tt_metal::IDevice* device) {
    CoreDescriptorSet dispatch_cores;
    unsigned num_cqs = device->num_hw_cqs();
    const auto& dispatch_core_config = tt::tt_metal::dispatch_core_manager::instance().get_dispatch_core_config();
    CoreType dispatch_core_type = dispatch_core_config.get_core_type();
    tt::log_warning("Dispatch Core Type = {}", dispatch_core_type);
    for (auto logical_core : tt::get_logical_dispatch_cores(device->id(), num_cqs, dispatch_core_config)) {
        dispatch_cores.insert({logical_core, dispatch_core_type});
    }
    return dispatch_cores;
}

inline uint64_t GetDprintBufAddr(tt::tt_metal::IDevice* device, const CoreCoord& virtual_core, int risc_id) {
    dprint_buf_msg_t* buf =
        device->get_dev_addr<dprint_buf_msg_t*>(virtual_core, tt::tt_metal::HalL1MemAddrType::DPRINT);
    return reinterpret_cast<uint64_t>(&(buf->data[risc_id]));
}

// TODO(#17275): Move this and others to the HAL
#define DPRINT_NRISCVS 5
#define DPRINT_NRISCVS_ETH 1

inline int GetNumRiscs(tt::tt_metal::IDevice* device, const CoreDescriptor& core) {
    if (core.type == CoreType::ETH) {
        return (device->arch() == tt::ARCH::BLACKHOLE)? DPRINT_NRISCVS_ETH + 1 : DPRINT_NRISCVS_ETH;
    } else {
        return DPRINT_NRISCVS;
    }
}

inline const std::string_view get_core_type_name(CoreType ct) {
    switch (ct) {
        case CoreType::ARC: return "ARC";
        case CoreType::DRAM: return "DRAM";
        case CoreType::ETH: return "ethernet";
        case CoreType::PCIE: return "PCIE";
        case CoreType::WORKER: return "worker";
        case CoreType::HARVESTED: return "harvested";
        case CoreType::ROUTER_ONLY: return "router_only";
        case CoreType::ACTIVE_ETH: return "active_eth";
        case CoreType::IDLE_ETH: return "idle_eth";
        case CoreType::TENSIX: return "tensix";
        default: return "UNKNOWN";
    }
}

}  // namespace tt::tt_metal

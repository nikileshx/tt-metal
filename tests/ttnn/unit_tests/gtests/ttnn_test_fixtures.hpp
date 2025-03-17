// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <math.h>
#include <algorithm>
#include <functional>
#include <random>

#include "gtest/gtest.h"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "tests/tt_metal/test_utils/env_vars.hpp"

#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_impl.hpp"
#include "hostdevcommon/common_values.hpp"

using namespace tt::tt_metal;  // For test

namespace ttnn {

class TTNNFixtureWithDevice : public ::testing::Test {
protected:
    tt::tt_metal::IDevice* device_ = nullptr;
    tt::ARCH arch_ = tt::ARCH::Invalid;
    size_t num_devices_ = 0;

    void SetUp() override {
        std::srand(0);
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        device_ = tt::tt_metal::CreateDevice(/*device_id=*/0);
    }

    void TearDown() override { tt::tt_metal::CloseDevice(device_); }
};

// TODO: deduplicate the code with `TTNNFixtureWithDevice`.
class MultiCommandQueueSingleDeviceFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi CQ test suite, since it can only be run in Fast Dispatch Mode.";
        }

        DispatchCoreType dispatch_core_type = DispatchCoreType::WORKER;
        if (arch_ == tt::ARCH::WORMHOLE_B0 and num_devices_ != 1) {
            tt::log_warning(
                tt::LogTest, "Ethernet Dispatch not being explicitly used. Set this configuration in Setup()");
            dispatch_core_type = DispatchCoreType::ETH;
        }
        device_ = tt::tt_metal::CreateDevice(
            0, 2, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, DispatchCoreConfig{dispatch_core_type});
    }

    void TearDown() override { tt::tt_metal::CloseDevice(device_); }

    tt::tt_metal::IDevice* device_;
    tt::ARCH arch_;
    size_t num_devices_;
};

// TODO: deduplicate the code with `TTNNFixtureWithDevice`.
class MultiCommandQueueT3KFixture : public ::testing::Test {
protected:
    void SetUp() override {
        auto slow_dispatch = getenv("TT_METAL_SLOW_DISPATCH_MODE");
        arch_ = tt::get_arch_from_string(tt::test_utils::get_umd_arch_name());
        num_devices_ = tt::tt_metal::GetNumAvailableDevices();
        if (slow_dispatch) {
            GTEST_SKIP() << "Skipping Multi CQ test suite, since it can only be run in Fast Dispatch Mode.";
        }
        if (num_devices_ < 8 or arch_ != tt::ARCH::WORMHOLE_B0) {
            GTEST_SKIP() << "Skipping T3K Multi CQ test suite on non T3K machine.";
        }
        // Enable Ethernet Dispatch for Multi-CQ tests.

        devs = tt::tt_metal::detail::CreateDevices(
            {0, 1, 2, 3, 4, 5, 6, 7},
            2,
            DEFAULT_L1_SMALL_SIZE,
            DEFAULT_TRACE_REGION_SIZE,
            DispatchCoreConfig{DispatchCoreType::ETH});
    }

    void TearDown() override { tt::tt_metal::detail::CloseDevices(devs); }

    std::map<chip_id_t, tt::tt_metal::IDevice*> devs;
    tt::ARCH arch_;
    size_t num_devices_;
};

}  // namespace ttnn

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import time
import pytest
import torch
from models.utility_functions import is_grayskull
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report
from loguru import logger
from models.utility_functions import (
    enable_persistent_kernel_cache,
    disable_persistent_kernel_cache,
)

from torchvision import models
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor
from models.experimental.functional_alexnet.tt.ttnn_alexnet import TT_Alexnet
from models.perf.perf_utils import prep_perf_report


def get_expected_times(alexnet):
    return (24.05, 1.44)


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", ((1),))
@pytest.mark.parametrize("input_tensor", [torch.rand((1, 3, 224, 224))], ids=["input_tensor"])
def test_alexnet(device, input_tensor, batch_size):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict=state_dict)

    ttnn_input = input_tensor.permute((0, 2, 3, 1))

    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    durations = []
    for i in range(2):
        start = time.time()
        tt_model = TT_Alexnet(device, ttnn_input.shape, parameters)
        ttnn_output_tensor = tt_model(ttnn_input)
        ttnn_output_tensor = ttnn.from_device(ttnn_output_tensor)
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("alexnet")

    prep_perf_report(
        model_name="models/experimental/functional_alexnet",
        batch_size=batch_size,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [1, 253.4],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_alexnet(batch_size, expected_perf):
    subdir = "ttnn_alexnet"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_grayskull() else 234.04

    command = f"pytest tests/ttnn/integration_tests/alexnet/test_ttnn_alexnet.py"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_alexnet{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

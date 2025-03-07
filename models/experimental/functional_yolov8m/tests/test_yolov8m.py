# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import sys
import ttnn
import time
import torch
import pytest
import torch.nn as nn
from loguru import logger
from models.utility_functions import is_wormhole_b0
from models.perf.perf_utils import prep_perf_report
from models.experimental.functional_yolov8m.tt.ttnn_yolov8m import YOLOv8m
from models.experimental.functional_yolov8m.reference import yolov8m_utils
from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import custom_preprocessor
from models.utility_functions import enable_persistent_kernel_cache, disable_persistent_kernel_cache
from models.perf.device_perf_utils import run_device_perf, check_device_perf, prep_device_perf_report

try:
    sys.modules["ultralytics"] = yolov8m_utils
    sys.modules["ultralytics.nn.tasks"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.conv"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.block"] = yolov8m_utils
    sys.modules["ultralytics.nn.modules.head"] = yolov8m_utils

except KeyError:
    print("models.experimental.functional_yolov8m.reference.yolov8m_utils not found.")


class Ensemble(nn.ModuleList):
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.cat(y, 1)
        return y, None


def attempt_load(weights, map_location=None):
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        w = "models/experimental/functional_yolov8m/demo/yolov8m.pt"
        ckpt = torch.load(w, map_location=map_location)
        model.append(ckpt["ema" if ckpt.get("ema") else "model"].float().eval())
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True
        elif type(m) is nn.Upsample:
            m.recompute_scale_factor = None

    if len(model) == 1:
        return model[-1]
    else:
        for k in ["names", "stride"]:
            setattr(model, k, getattr(model[-1], k))
        return model


def get_expected_times(name):
    base = {"yolov8m": (177.47, 4.81)}
    return base[name]


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize("batch_size", [(8)])
@pytest.mark.parametrize("input_tensor", [torch.rand((8, 3, 320, 320))], ids=["input_tensor"])
def test_yolov8m(device, input_tensor, batch_size):
    disable_persistent_kernel_cache()

    torch_model = attempt_load("yolov8m.pt", map_location="cpu")

    state_dict = torch_model.state_dict()

    bs, inp_h, inp_w = input_tensor.shape[0], input_tensor.shape[2], input_tensor.shape[3]

    parameters = custom_preprocessor(device, state_dict, inp_h=inp_h, inp_w=inp_w)

    # core_grid = ttnn.CoreGrid(y=8, x=8)
    # n, c, h, w = input_tensor.shape

    # # sharded mem config for fold input
    # num_cores = core_grid.x * core_grid.y
    # shard_h = (n * w * h + num_cores - 1) // num_cores
    # grid_size = core_grid
    # grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    # shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    # shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
    # input_mem_config = ttnn.MemoryConfig(
    #     ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    # )
    # ttnn_input = input_tensor.permute(0, 2, 3, 1)
    # ttnn_input = ttnn_input.reshape(1, 1, h * w * n, c)
    # ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    # ttnn_input = ttnn.pad(ttnn_input, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

    # ttnn_input = ttnn_input.to(device, input_mem_config)

    # for batch size 8

    ttnn_input = input_tensor.permute((0, 2, 3, 1))
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    durations = []

    for i in range(2):
        start = time.time()
        ttnn_model_output = YOLOv8m(device, ttnn_input, parameters, res=(inp_h, inp_w), batch_size=bs)[0]
        end = time.time()
        durations.append(end - start)
        enable_persistent_kernel_cache()

    inference_and_compile_time, inference_time, *_ = durations

    expected_compile_time, expected_inference_time = get_expected_times("yolov8m")

    prep_perf_report(
        model_name="models/experimental/functional_yolov8m",
        batch_size=bs,
        inference_and_compile_time=inference_and_compile_time,
        inference_time=inference_time,
        expected_compile_time=expected_compile_time,
        expected_inference_time=expected_inference_time,
        comments="",
        inference_time_cpu=0.0,
    )

    logger.info(f"{durations}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Compile time: {inference_and_compile_time - inference_time}")
    logger.info(f"Inference time: {inference_time}")
    logger.info(f"Samples per second: {1 / inference_time * batch_size}")


@pytest.mark.parametrize(
    "batch_size, expected_perf",
    [
        [8, 342.42],
        # [8, 123.70],
    ],
)
@pytest.mark.models_device_performance_bare_metal
def test_perf_device_bare_metal_yolov8m(batch_size, expected_perf):
    subdir = "ttnn_yolov8m"
    num_iterations = 1
    margin = 0.03
    expected_perf = expected_perf if is_wormhole_b0() else 123.70

    command = f"pytest tests/ttnn/integration_tests/yolov8m/test_ttnn_yolov8m.py::test_demo"
    cols = ["DEVICE FW", "DEVICE KERNEL", "DEVICE BRISC KERNEL"]

    inference_time_key = "AVG DEVICE KERNEL SAMPLES/S"
    expected_perf_cols = {inference_time_key: expected_perf}

    post_processed_results = run_device_perf(command, subdir, num_iterations, cols, batch_size)
    expected_results = check_device_perf(post_processed_results, margin, expected_perf_cols)

    logger.info(f"{expected_results}")

    prep_device_perf_report(
        model_name=f"ttnn_functional_yolov8m{batch_size}",
        batch_size=batch_size,
        post_processed_results=post_processed_results,
        expected_results=expected_results,
        comments="",
    )

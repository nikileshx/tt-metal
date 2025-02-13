# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch, ttnn
from loguru import logger
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_alexnet.tt.ttnn_alexnet import TT_Alexnet
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [(torch.rand((12, 3, 224, 224)))],
    ids=["input_tensor1"],
)
def test_alexnet(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    state_dict = torch_model.state_dict()
    parameters = custom_preprocessor(device, state_dict=state_dict)

    core_grid = ttnn.CoreGrid(y=8, x=8)
    n, c, h, w = input_tensor.shape

    # sharded mem config for fold input
    num_cores = core_grid.x * core_grid.y
    shard_h = (n * w * h + num_cores - 1) // num_cores
    grid_size = core_grid
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (shard_h, 16), ttnn.ShardOrientation.ROW_MAJOR)
    input_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    ttnn_input = input_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn_input.reshape(1, 1, h * w * n, c)
    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_input = ttnn.pad(ttnn_input, [1, 1, n * h * w, 16], [0, 0, 0, 0], 0)

    ttnn_input = ttnn_input.to(device, input_mem_config)

    with torch.inference_mode():
        tt_model = TT_Alexnet(device, input_tensor.shape, parameters)
        ttnn_output_tensor = tt_model(ttnn_input)

        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    with torch.inference_mode():
        torch_output_tensor = torch_model(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

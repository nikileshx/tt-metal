# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch.nn as nn
import torch, ttnn
from loguru import logger
from torchvision import models
from models.utility_functions import is_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.experimental.functional_alexnet.tt.ttnn_alexnet import TT_Alexnet
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor


def get_mesh_mappers(device):
    is_mesh_device = isinstance(device, ttnn.MeshDevice)
    if is_mesh_device:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, output_mesh_composer


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "batch_size",
    [32],
    ids=["bs1"],
)
def test_alexnet(mesh_device, batch_size):
    device = mesh_device

    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    state_dict = torch_model.state_dict()

    mesh_device_flag = is_wormhole_b0() and ttnn.GetNumAvailableDevices() == 2
    batch_size = (2 * batch_size) if mesh_device_flag else batch_size

    input_tensor = torch.rand((batch_size, 3, 224, 224))

    inputs_mesh_mapper, output_mesh_composer = get_mesh_mappers(device)

    parameters = custom_preprocessor(device, state_dict=state_dict)

    ttnn_input = input_tensor.permute(0, 2, 3, 1)
    ttnn_input = ttnn.from_torch(
        ttnn_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=inputs_mesh_mapper,
        device=device,
    )

    with torch.inference_mode():
        tt_model = TT_Alexnet(device, ttnn_input.shape, parameters, inputs_mesh_mapper, output_mesh_composer)
        ttnn_output_tensor = tt_model(ttnn_input)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor, mesh_composer=output_mesh_composer)

    with torch.inference_mode():
        torch_output_tensor = torch_model(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

    logger.info(f"Inference done for batch size: {ttnn_output_tensor.shape[0]}")

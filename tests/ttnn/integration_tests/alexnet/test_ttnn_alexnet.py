# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch, ttnn
from loguru import logger
from torchvision import models
from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.model_preprocessing import preprocess_model_parameters
from models.experimental.functional_alexnet.tt.ttnn_alexnet import ttnn_alexnet
from models.utility_functions import disable_persistent_kernel_cache
from models.experimental.functional_alexnet.tt.ttnn_alexnet_utils import custom_preprocessor


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_tensor",
    [(torch.rand((1, 3, 224, 224)))],
    ids=["input_tensor1"],
)
def test_alexnet(device, input_tensor):
    disable_persistent_kernel_cache()

    torch_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    torch_model.eval()

    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_model,
        convert_to_ttnn=lambda *_: True,
        device=device,
        custom_preprocessor=custom_preprocessor,
    )

    ttnn_input = input_tensor.permute((0, 2, 3, 1))

    ttnn_input = ttnn.from_torch(ttnn_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with torch.inference_mode():
        ttnn_output_tensor = ttnn_alexnet(device, ttnn_input, parameters)
        ttnn_output_tensor = ttnn.to_torch(ttnn_output_tensor)

    with torch.inference_mode():
        torch_output_tensor = torch_model(input_tensor)

    passing, pcc = assert_with_pcc(ttnn_output_tensor, torch_output_tensor, 0.99)
    logger.info(f"Passing: {passing}, PCC: {pcc}")

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import random
import pytest
import torch
import ttnn
import traceback

from tests.ttnn.python_api_testing.sweep_tests import ttnn_ops
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


def run_eltwise_sqrt_tests(
    input_shape,
    dtype,
    dlayout,
    in_mem_config,
    output_mem_config,
    data_seed,
    device,
):
    torch.manual_seed(data_seed)
    x = torch.Tensor(size=input_shape[0]).uniform_(1, 100).to(torch.bfloat16)  # Range will be updated after #14077.

    try:
        # get ref result
        ref_value = torch.sqrt(x)
        logger.info(f"PyTorch: {ref_value[0:10, 0:10]}")

        x = ttnn_ops.setup_ttnn_tensor(x, device, dlayout[0], in_mem_config[0], dtype[0])

        tt_result = ttnn.sqrt(x)
        tt_result = ttnn_ops.ttnn_tensor_to_torch(tt_result, output_mem_config)
        logger.info(f"Tenstorrent: {tt_result[0:10, 0:10]}")

    except Exception as e:
        logger.warning(f"Test execution crashed: {e}")
        print(traceback.format_exc())
        raise e

    assert len(tt_result.shape) == len(ref_value.shape)
    assert tt_result.shape == ref_value.shape

    # compare tt and golden outputs
    success, pcc_value = comp_pcc(ref_value, tt_result)
    logger.debug(pcc_value)
    logger.debug(success)

    assert success


test_sweep_args = [
    (
        [(160, 128)],
        [ttnn.bfloat8_b],
        [ttnn.TILE_LAYOUT],
        [ttnn.L1_MEMORY_CONFIG],
        ttnn.L1_MEMORY_CONFIG,
        13382297,
    ),
]


@pytest.mark.parametrize(
    "input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed",
    (test_sweep_args),
)
def test_eltwise_sqrt(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device):
    run_eltwise_sqrt_tests(input_shape, dtype, dlayout, in_mem_config, out_mem_config, data_seed, device)

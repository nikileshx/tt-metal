import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from loguru import logger


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768}], indirect=True)
@pytest.mark.parametrize(
    "input_specs",
    [[(1, 1, 400, 32), (1, 1, 400, 32)]],
    #  [[(1, 1, 1600, 192), (1, 1, 1600, 384)]],
    # [[(1, 1, 1600, 400), (1, 1, 1600, 576)]],
    ids=[
        "input_spec1",
    ],
)
def test_sharded_concat(
    device, input_specs, num_cores=64, dim=3
):  # expected input tensors to be in fp16, RM, same (h*w)
    input_tensors = []
    torch_input_tensors = []
    for shape in input_specs:
        torch_input_tensor = torch.rand(shape, dtype=torch.bfloat16)
        torch_input_tensors.append(torch_input_tensor)
        input_tensors.append(ttnn.from_torch(torch_input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT, device=device))

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})

    input_sharded_memory_config = []

    for i in range(len(input_tensors)):
        in_shard_width = input_tensors[i].shape[-1]
        shard_height = (input_tensors[i].shape[2] + num_cores - 1) // num_cores
        memory_config = ttnn.create_sharded_memory_config(
            (shard_height, in_shard_width),
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_config.append(memory_config)

    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config[i])

    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.concat(torch_input_tensors, dim=dim)

    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)
    print(output)
    print(output.memory_config())
    output = ttnn.to_torch(output)

    logger.info(f"{assert_with_pcc(torch_output_tensor, output)}")


def get_concat_shard(device, input_specs, num_cores=64, dim=3):
    input_sharded_memory_config = []

    for i in range(len(input_tensors)):
        in_shard_width = input_tensors[i].shape[-1]
        shard_height = (input_tensors[i].shape[2] + num_cores - 1) // num_cores
        memory_config = ttnn.create_sharded_memory_config(
            (shard_height, in_shard_width),
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_config.append(memory_config)

    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config[i])

    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )

    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)

    return output

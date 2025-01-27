import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

torch.manual_seed(0)
device_id = 0
device = ttnn.open_device(device_id=device_id)
ttnn.enable_program_cache(device)


def custom_meshgrid(*tensors, indexing="ij"):
    if not tensors:
        raise ValueError("meshgrid expects a non-empty list of tensors")
    size = len(tensors)

    if indexing == "xy" and size >= 2:
        tensors = list(tensors)
        tensors[0], tensors[1] = tensors[1], tensors[0]
    elif indexing not in ("ij", "xy"):
        raise ValueError("indexing must be one of 'ij' or 'xy'")

    shape = [t.numel() for t in tensors]
    grids = []
    for i, t in enumerate(tensors):
        ttnn_tensor = ttnn.from_torch(t)
        ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout=ttnn.TILE_LAYOUT)

        reshape_1 = [1] * (size + 3)  # Add extra dimensions padded for safety
        reshape_1[i] = shape[i]  # Expand only along current tensor's axis

        repeat_shape = [1] * (size + 3)
        for j in range(size):
            repeat_shape[j] = shape[j] if j != i else 1  # Repeat along other axes

        reshape_2 = shape[:size]  # Reshape back to original dimensions

        ttnn_tensor = ttnn.reshape(ttnn_tensor, tuple(reshape_1))
        ttnn_tensor = ttnn.repeat(ttnn_tensor, ttnn.Shape(repeat_shape[: len(reshape_1)]))
        ttnn_tensor = ttnn.reshape(ttnn_tensor, tuple(reshape_2))

        expanded = ttnn.to_torch(ttnn_tensor)
        grids.append(expanded)

    if indexing == "xy" and size >= 2:
        grids[0], grids[1] = grids[1], grids[0]
    return grids


@pytest.mark.parametrize(
    "shapes",
    [
        [(2,), (2,), (2,)],
        [(32,), (1,), (32,), (32,), (32,)],  # 1D and 1D
        [(1,), (64,), (5,), (2,), (12,)],  # 1D and 1D
        [(32,), (16,), (8,), (4,)],  # Gradually reducing sizes
        [(7,), (1,), (5,), (3,), (9,)],  # Irregular sizes
        [(50,), (25,), (5,), (2,), (1,)],  # Larger ranges with decreasing sizes
        [(2,), (2,), (2,), (2,), (2,), (2,)],  # Multiple tensors of the same small size
        # Edge cases
        [(1,)],
        [(2,), (1,), (2,)],
        [(2,), (2,), (32,), (16,), (8,), (4,), (2,), (1,)],
    ],
)
def test_custom_meshgrid(shapes, device):
    # print(shapes)
    # torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in shapes]
    torch_tensors = [
        torch.tensor([1, 2, 3], dtype=torch.bfloat16),
        torch.tensor([4, 5, 6], dtype=torch.bfloat16),
    ]  # , torch.tensor([4, 5], dtype=torch.bfloat16), torch.tensor([4, 5, 6], dtype=torch.bfloat16)

    ttnn_tensors = [ttnn.from_torch(tensor) for tensor in torch_tensors]
    ttnn_tensors = [ttnn.to_device(tensor, device, memory_config=ttnn.L1_MEMORY_CONFIG) for tensor in ttnn_tensors]
    ttnn_tensors = [ttnn.to_layout(tensor, ttnn.TILE_LAYOUT) for tensor in ttnn_tensors]

    torch_grids = torch.meshgrid(torch_tensors, indexing="ij")
    print("Expected:", torch_grids)
    print("result.shape:", torch_grids[0].shape)
    ttnn_grids = custom_meshgrid(*torch_tensors)

    for ttnn_grid, torch_grid in zip(ttnn_grids, torch_grids):
        assert_with_pcc(ttnn_grid, torch_grid)


ttnn.close_device(device)

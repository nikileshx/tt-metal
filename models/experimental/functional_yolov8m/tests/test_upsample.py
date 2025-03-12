import torch

import torch.nn as nn

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout
from loguru import logger

device = ttnn.CreateDevice(0)

torch.manual_seed(0)

scale_factor = (1, 4)

input = torch.randn(4, 10, 10, 256, dtype=torch.bfloat16)

input_tensor = ttnn.from_torch(input, device=device)

output_tensor = ttnn.upsample(input_tensor, scale_factor, memory_config=ttnn.L1_MEMORY_CONFIG)
output_tensor_1 = ttnn.to_torch(output_tensor)
print(output_tensor_1.shape)
output_tensor_1 = output_tensor_1.reshape((4, 20, 20, 256))
print(f"output_tensor1 after reshape: {output_tensor_1}")


# print("Input >>>>>>>>>>>>>>>")
# print(input_tensor)
# print("<<<<<<<<<<<<<<<<<<<<<<")
# print("Output Tensor 1>>>>>>>>>>>>>..\n",output_tensor)

# print("<<<<<<<<<<<<<<<<<<<<<<<<")

input = input.reshape((1, 1, 800, 576))  # 1, 2, 800, 256 # 1,1,1600, 256

input_tensor = ttnn.from_torch(input, device=device)
# print("Input >>>>>>>>>>>>>>>")
# print(input_tensor)
# print("<<<<<<<<<<<<<<<<<<<<<<")


output_tensor = ttnn.upsample(input_tensor, (1, 4), memory_config=ttnn.L1_MEMORY_CONFIG, conv_out_shape=False)
output_tensor_2 = ttnn.to_torch(output_tensor)
output_tensor_2 = output_tensor_2.reshape((4, 20, 20, 256))


print("output_tensor_2>>>>>>>>>>>>>>>>>>>>>\n", output_tensor_2)


print("<<<<<<<<<<<<<<<<<<<<<<<<")

logger.info(assert_with_pcc(output_tensor_2, output_tensor_1))

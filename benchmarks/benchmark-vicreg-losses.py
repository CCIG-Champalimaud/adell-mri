import os
import sys
import time
from tqdm import trange

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from adell_mri.modules.self_supervised import VICRegLoss, VICRegLocalLoss

batch_size = 32
N = 100

input_boxes_a = torch.randint(16, size=[batch_size, 6])
input_boxes_a[:, 3:] = input_boxes_a[:, :3] + torch.tensor([4, 4, 2])
input_boxes_b = torch.randint(16, size=[batch_size, 6])
input_boxes_b[:, 3:] = input_boxes_b[:, :3] + torch.tensor([4, 4, 2])

input_boxes_a = input_boxes_a.float().to("cuda")
input_boxes_b = input_boxes_b.float().to("cuda")

for f in [16, 32, 64, 128, 512, 1024]:
    print("Running VICreg with {} features".format(f))
    input_tensor_a = torch.rand(size=(batch_size, f, 4, 4, 2)).to("cuda")
    input_tensor_b = torch.rand(size=(batch_size, f, 4, 4, 2)).to("cuda")

    times = 0
    L = VICRegLoss()
    for _ in trange(N):
        a = time.time()
        L(input_tensor_a, input_tensor_b)
        b = time.time()
        times += b - a
    print("Average time for VICReg ({} features): {}s".format(f, times / N))

    print("Running VICRegLocal with {} features".format(f))

    times = 0
    L = VICRegLocalLoss()
    for _ in trange(N):
        a = time.time()
        L(input_tensor_a, input_tensor_b, input_boxes_a, input_boxes_b)
        b = time.time()
        times += b - a
    print(
        "Average time for VICRegLocal ({} features): {}s".format(f, times / N)
    )

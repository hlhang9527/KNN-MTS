import os
import numpy as np
import torch

def get_np_memmap(data_dir, file_name, shape, dtype=np.float, mode="r"):
    file_name = os.path.join(data_dir, file_name)
    print("get np memap from: {}".format(file_name))
    np_memmap = np.memmap(file_name, shape=shape, dtype=dtype, mode=mode)
    return np_memmap

def  batch_cosine_similarity(x, y):
    # x: bxnxd
    # y: bxnxd
    # 计算分母
    l2_x = torch.norm(x, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_y = torch.norm(y, dim=2, p=2) + 1e-7  # avoid 0, l2 norm, num_heads x batch_size x hidden_dim==>num_heads x batch_size
    l2_m = torch.matmul(l2_x.unsqueeze(dim=2), l2_y.unsqueeze(dim=2).transpose(1, 2))
    # 计算分子
    l2_z = torch.matmul(x, y.transpose(1, 2))
    # cos similarity affinity matrix
    cos_affnity = l2_z / l2_m
    adj = cos_affnity
    return adj
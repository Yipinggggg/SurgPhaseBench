from torch.distributions.normal import Normal
from torch.distributions.chi2 import Chi2
import torch
import numpy as np


def class2boundary(target):
    """
    :param target: (B, L)
    :return: (B, L), begin:0 , middle:1, end:2
    """
    target = target.cpu().numpy()
    B, T = target.shape
    assert B == 1

    boundary = np.zeros((B, T), dtype=np.int64)
    begin_list = []
    end_list = []

    for b in range(B):
        boundary[b, 0] = 0
        for t in range(1, T - 1):
            if target[b, t] != target[b, t - 1]:
                boundary[b, t] = 0
                begin_list.append(t)
            else:
                if target[b, t] == target[b, t + 1]:
                    boundary[b, t] = 1
                else:
                    boundary[b, t] = 2
                    end_list.append(t)
        boundary[b, T - 1] = 2
    # print(boundary)
    boundary = torch.from_numpy(boundary)
    begin = torch.tensor(begin_list)
    end = torch.tensor(end_list)
    return boundary, begin, end


def create_distribution_from_cls(cls, window_size, chi2=True):
    """
    cls: 0 is begin, 1 is middle, 2 is end
    return (window_size)
    """
    if chi2:
        if cls == 0:
            dis = create_chi2_distribution(window_size, right=True)
        elif cls == 2:
            dis = create_chi2_distribution(window_size, right=False)
        else:
            dis = create_normal_distribution(10, window_size)
    else:
        if cls == 0:
            dis = create_half_distribution(window_size, right=True)
        elif cls == 2:
            dis = create_half_distribution(window_size, right=False)
        else:
            dis = create_normal_distribution(10, window_size)
    return dis


def create_normal_distribution(scale, window_size):
    norm_distribution = Normal(0, scale)
    point_index = torch.arange(-(window_size//2), window_size//2+1)
    dis = torch.exp(norm_distribution.log_prob(point_index))
    dis = dis / torch.sum(dis)
    return dis


def create_uniform_distribution(window_size):
    return torch.ones(window_size) / window_size


def create_chi2_distribution(window_size, right=True):
    """
    right: right part has higher similarity
    """
    chi2_distribution = Chi2(4)
    point_index = 2.0 / ((window_size+1)//2) * torch.arange(-(window_size//2), window_size//2+1) + 2
    dis = torch.exp(chi2_distribution.log_prob(point_index))
    dis = dis / torch.sum(dis)
    if right:
        return dis
    else:
        return dis[range(window_size-1, -1, -1)]


def create_half_distribution(window_size, right=True):
    """
    right: right part has higher similarity
    """
    dis = torch.zeros(window_size)
    dis[-window_size//2:] = 1 / (window_size//2+1)
    if right:
        return dis
    else:
        return dis[range(window_size-1, -1, -1)]


def extract_dis_from_attention(attention_map, window_size):
    """
    :param attention_map: [B, H, L_seg , (L_seg + 2 * window_size // 2)] or [B, H, L, L]
    :return: [B, H, L_seg, window_size]
    """
    B, H, l1, l2 = attention_map.shape
    if l1 == l2:
        attention_map = torch.cat([torch.zeros(B, H, l1, window_size // 2, device=attention_map.device),
                                   attention_map,
                                   torch.zeros(B, H, l1, window_size // 2, device=attention_map.device)], dim=3)
        l2 = attention_map.shape[3]

    assert attention_map.shape[2] + 2 * (window_size // 2) == attention_map.shape[3]
    dis =  torch.as_strided(attention_map, (B, H, l1, window_size), (H*l1*l2, l1*l2, l2+1, 1))
    return dis


def KL_loss(scores_dis, dis):
    """
    Kullback Leibler Divergence
    :param scores_dis: (B, H, L, window_size)
    :param dis: (window_size)
    :return: scalar
    """
    kl = dis * (torch.log(dis + 0.00001) - torch.log(scores_dis + 0.00001))
    return torch.mean(torch.sum(kl, dim=-1))

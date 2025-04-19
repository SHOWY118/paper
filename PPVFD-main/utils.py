"""
Reference:
https://github.com/FedML-AI/FedML
"""

import numpy as np
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.nn import functional as F
import torch
import random

class RunningAverage():
    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total = self.total+val
        self.steps = self.steps+1

    def value(self):
        return self.total / float(self.steps)


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



class KL_Loss(nn.Module):
    def __init__(self, temperature=3.0):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)
        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)
        return loss


class CE_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(CE_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1)
        loss = -self.T * self.T * torch.sum(torch.mul(output_batch, teacher_outputs)) / teacher_outputs.size(0)
        return loss


def mal_select_clients(N, r, seed=42):
    random.seed(seed)  # 固定随机种子
    num_selected_clients = int(N * r)  # 计算要选择的客户端数量
    all_clients = list(range(N))  # 客户端索引列表
    selected_clients = random.sample(all_clients, num_selected_clients)  # 随机选出客户端
    return selected_clients


#logits中毒攻击相关的
#顺移一位
def change_logit(logit):
    # 将一维张量转换为二维张量
    logit = logit.unsqueeze(0)
    num_columns = logit.size(1)
    # 使用 torch.topk 获取每一行的值和索引
    values, indices = torch.topk(logit, k=num_columns, dim=1)
    changed_indices = torch.cat((indices[:, 1:], indices[:, 0].unsqueeze(1)), dim=1)

    values = values.to(logit)
    logit = logit.scatter_(dim=1, index=changed_indices, src=values)
    # 输出已修改的张量output
    return logit.squeeze(0)


def repalceLogitsWith0(log_probs):
    # 将所有 logits 设置为0
    log_probs = log_probs.zero_()
    return log_probs

def replace_logits_with_random(log_probs):
    # 指定均值和方差
    mean = 0.0
    std = 1.0
    # 生成具有指定均值和方差的随机数
    random_logits = torch.normal(mean=mean, std=std, size=log_probs.size())

    # 将输入的 logits 替换为随机数
    log_probs.data.copy_(random_logits)

    return log_probs


def replace_logit_second_with_C(log_probs,C):
    # logit = logit.detach()
    # 将一维张量转换为二维张量
    logit = log_probs.unsqueeze(0)
    # 找到第二高的置信度及其索引
    values, indices = torch.topk(logit, k=2, dim=1)

    # 创建索引张量
    logit.fill_(-C)

    # 将第二高置信度的位置设置为9
    logit.scatter_(1, indices[:, 1:], C)

    # 输出已修改的张量output
    return logit.squeeze(0)


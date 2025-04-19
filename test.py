import torch
import torch.nn.functional as F

# 定义 change_logit 函数
def change_logit(logit):
    logit = logit.unsqueeze(0)  # [1, N]
    num_columns = logit.size(1)
    values, indices = torch.topk(logit, k=num_columns, dim=1)
    changed_indices = torch.cat((indices[:, 1:], indices[:, 0].unsqueeze(1)), dim=1)
    values = values.to(logit.dtype)
    logit = logit.scatter_(dim=1, index=changed_indices, src=values)
    return logit.squeeze(0)


def FDPLA_logits(log_probs):
    # 获取 log_probs 的维度
    num_rows, num_columns = log_probs.shape

    # 创建一个与 log_probs 形状相同的 tensor 来存储结果
    changed_logits = torch.empty_like(log_probs)

    for i in range(num_rows):
        # 获取升序和降序的排序索引
        ascending_indices = torch.argsort(log_probs[i], descending=False)  # 升序排序的索引
        descending_indices = torch.argsort(log_probs[i], descending=True)  # 降序排序的索引

        # 按照升序和降序索引取值
        sorted_ascending = log_probs[i][ascending_indices]  # 最小值排在前面
        sorted_descending = log_probs[i][descending_indices]  # 最大值排在前面

        # 重新排列，将最大的值对应最小值，依次类推
        # 使用 scatter_ 函数将升序的值放置到降序的索引位置
        changed_logits[i].scatter_(0, descending_indices, sorted_ascending)

    return changed_logits


def FDLA_logits(log_probs):
    num_rows, num_columns = log_probs.shape
    changed_logits = torch.empty_like(log_probs)  # 创建与 log_probs 相同形状的 tensor
    for i in range(num_rows):
        sorted_indices = torch.argsort(log_probs[i], descending=True)  # 获取降序排序的索引
        sorted_values = log_probs[i][sorted_indices]  # 按照降序索引取值
        changed_indices = torch.cat((sorted_indices[1:], sorted_indices[:1]))  # 改变索引顺序

        # 使用 scatter_ 来进行索引赋值，避免潜在的索引赋值问题
        changed_logits[i].scatter_(0, changed_indices, sorted_values)

    return changed_logits

def FDSA_logits(log_probs):
    """
    实现 FDSA 攻击：值对换、均值回归、放缩
    :param log_probs: 输入 logits，形状 [batch_size, num_classes]
    :param regression_factor: 均值回归的调整比例 (0 到 1)，控制趋近均值的程度
    :param scale_factor: 放缩倍率，控制整体放大/缩小
    :return: 攻击后的 logits
    """
    # 复制输入以避免修改原始数据
    regression_factor = 0.5
    scale_factor = 2.0
    attacked_logits = log_probs.clone()
    batch_size, num_classes = attacked_logits.shape

    for i in range(batch_size):
        # 当前行的 logits
        row = attacked_logits[i]

        # 步骤 1：值对换
        # 找到最大值和次大值的索引
        values, indices = torch.topk(row, 2)  # 获取最大的两个值及其索引
        max_val, second_max_val = values[0], values[1]
        max_idx, second_max_idx = indices[0], indices[1]

        # 交换最大值和次大值
        row[max_idx] = second_max_val
        row[second_max_idx] = max_val

        # 步骤 2：均值回归
        # 计算最大值和次大值的均值
        mean_val = (row[max_idx] + row[second_max_idx]) / 2

        # 调整最大值（减小）和次大值（增大）向均值靠拢
        row[max_idx] = row[max_idx] + regression_factor * (mean_val - row[max_idx])
        row[second_max_idx] = row[second_max_idx] + regression_factor * (mean_val - row[second_max_idx])

        # 更新行
        attacked_logits[i] = row

    # 步骤 3：放缩
    attacked_logits = attacked_logits * scale_factor

    return attacked_logits


# 假设有两个一维 logits 向量
logit1 = torch.tensor([[0.2, 0.4, 0.1, 0.5],[0.3, 0.6, 0.2, 0.4]])
# logit2 = torch.tensor([0.3, 0.6, 0.2, 0.4])
logit2 = torch.tensor([-0.2, -0.4, -0.1, -0.5])

# 计算原始 logits 向量之间的余弦相似度
original_cosine_similarity = F.cosine_similarity(logit1.unsqueeze(0), logit2.unsqueeze(0))

# 对 logits 进行调整
changed_logit1 = FDLA_logits(logit1.cpu().detach())
changed_logit2 = change_logit(logit2.cpu().detach())
changed_logit3= FDSA_logits(logit1.cpu().detach())
print("logit1:{}".format(changed_logit1))
print("logit2:{}".format(changed_logit2))
print("logit3:{}".format(changed_logit3))




# import numpy as np
#
# def replace_second_with_one(log_probs):
#     # 找到每一行中第二高的置信度及其索引
#     values = np.sort(log_probs, axis=1)[:, -2:]  # 获取每行的前两个最大值
#     indices = np.argsort(log_probs, axis=1)  # 获取原始索引
#
#     # 创建一个全零数组作为输出
#     output = np.zeros_like(log_probs)
#
#     # 将第二高置信度的位置设置为1
#     for i in range(log_probs.shape[0]):
#         second_highest_index = indices[i, -2]  # 获取第二高置信度的索引
#         output[i, second_highest_index] = 1  # 将对应位置设置为1
#
#     return output
#
#
# def FDLA_logits( logits):
#     num_rows, num_columns = logits.shape
#     changed_logits = np.empty_like(logits)
#     for i in range(num_rows):
#         sorted_indices = np.argsort(logits[i])[::-1]  # 获取降序排序的索引
#         sorted_values = logits[i][sorted_indices]  # 按照降序索引取值
#         changed_indices = np.concatenate([sorted_indices[1:], sorted_indices[:1]])
#         changed_logits[i, changed_indices] = sorted_values
#     return changed_logits
#
# # 示例使用
# logits = np.array([[0.1, 0.5, 0.3],
#                    [0.4, 0.2, 0.9]])
# # modified_logits = replace_second_with_one(logits)
# modified_logits =FDLA_logits(logits)
# A=change_logit(torch.tensor(logits[1]))
# print(modified_logits,A)

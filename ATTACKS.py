import numpy as np
import torch
import torch.nn as nn

from torch.nn import functional as F
import torch
class LogitsProcessor:
    def __init__(self,attack_method_id,global_knowledge):
        self.attack_method_id=attack_method_id
        self.global_knowledge=global_knowledge

    def attack(self, log_probs):
        if self.attack_method_id == 1:
            # print("执行FDLA攻击")
            log_probs= self.FDLA_logits(log_probs)
        elif self.attack_method_id == 2:
            # print("执行PCFDLA攻击")
            log_probs=self.PCFDLA_logits(log_probs)
        elif self.attack_method_id == 3:
            # print("执行FDPLA攻击")
            log_probs= self.FDPLA_logits(log_probs)
        elif self.attack_method_id == 4:
            # print("执行Fixed攻击")
            log_probs= self.Fixed_logits(log_probs)
        elif self.attack_method_id == 5:
            # print("执行Random攻击")
            log_probs= self.Random_logits(log_probs)
        elif self.attack_method_id == 9:
            log_probs=self.FDSA_logits(log_probs)
        return log_probs

    def FDSA_logits(self, log_probs):
        """
        实现 FDSA 攻击：值对换、均值回归、放缩
        :param log_probs: 输入 logits，形状 [batch_size, num_classes]
        :param regression_factor: 均值回归的调整比例 (0 到 1)，控制趋近均值的程度
        :param scale_factor: 放缩倍率，控制整体放大/缩小
        :return: 攻击后的 logits
        """
        # 复制输入以避免修改原始数据
        regression_factor = 0.5
        mean_vector_glo = np.linalg.norm(np.mean(self.global_knowledge, axis=0))
        mean_vector_loc = torch.norm(log_probs.mean(dim=0),p=2)
        scale_factor = max(2*mean_vector_glo/mean_vector_loc,2)
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


    def FDLA_logits(self, log_probs):
        num_rows, num_columns = log_probs.shape
        changed_logits = torch.empty_like(log_probs)
        for i in range(num_rows):
            sorted_indices = torch.argsort(log_probs[i], descending=True)  # 获取降序排序的索引
            sorted_values = log_probs[i][sorted_indices]  # 按照降序索引取值
            changed_indices = torch.cat((sorted_indices[1:], sorted_indices[:1]))  # 改变索引顺序
            changed_logits[i].scatter_(0, changed_indices, sorted_values)
        return changed_logits

    def FDPLA_logits(self, log_probs):
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

    def PCFDLA_logits(self, log_probs):
        s = torch.FloatTensor(1).uniform_(-20, 20).cuda().requires_grad_()
        indices = torch.argsort(log_probs, dim=1)  # 获取原始索引，按照行排序
        output = -s * torch.ones_like(log_probs)  # 创建与 log_probs 相同形状的输出张量，初始值为 -s
        for i in range(log_probs.size(0)):  # 遍历每一行
            second_highest_index = indices[i, -2]  # 获取第二高置信度的索引
            output[i, second_highest_index] = s  # 将对应位置设置为 s
        return output

    def Fixed_logits(self, log_probs):
        num_columns = log_probs.size(1)
        random_values = torch.FloatTensor(num_columns).uniform_(-20, 20)
        for i in range(num_columns):
            log_probs[:, i] = random_values[i]
        return log_probs

    def Random_logits(self, log_probs):
        mean = 1.0
        std = 10.0
        # 生成与 log_probs 相同形状的随机张量
        random_logits = torch.normal(mean=mean, std=std, size=log_probs.size())
        log_probs.data.copy_(random_logits)
        return log_probs

class IPMAttack:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def attack(self, w_locals, mal_idxs, device):
        for name in w_locals[0]:
            users_grads = []
            if 'conv' in name or 'fc' in name:
                # print("层：{}翻转成功".format(name))
                for i in range(len(w_locals)):
                    if i in mal_idxs:
                        continue
                    users_grads.append(w_locals[i][name].detach())
                grads_mean = torch.mean(torch.stack(users_grads), dim=0)
                mal_grads = self._attack_grads(grads_mean)
                for mal_idx in mal_idxs:
                    w_locals[mal_idx][name] = mal_grads.clone().to(device)

    def _attack_grads(self, grads_mean):
        if grads_mean.ndim > 0:
            grads_mean[:] = - self.epsilon * grads_mean[:]
        else:
            grads_mean = - self.epsilon * grads_mean
        return grads_mean

class ParameterFlipAttack:
    def __init__(self):
        pass
    def attack(self, w_locals, mal_idxs, device):
        for name in w_locals[0]:
            # print("name:{}".format(name))
            if 'conv' in name or 'fc' in name:
            # 对于每个被攻击的客户端，执行参数符号反转
                for mal_idx in mal_idxs:
                    # 获取当前参数
                    param = w_locals[mal_idx][name]
                    # 将参数符号反转
                    flipped_param = -param.detach()
                    # 更新被攻击客户端的参数
                    w_locals[mal_idx][name] = flipped_param.to(device)  # 将更新后的参数移回 GPU

class LabelFlipAttack:
    def __init__(self, class_num):
        self.class_num = class_num

    def attack(self, client_index, labels):
        # 检查是否是恶意客户端
            # 标签翻转：将第 y 类标签翻转为 C-y 类标签
        flipped_labels = torch.tensor([self.class_num - 1 - label.item() for label in labels],
                                           dtype=torch.long).cuda()
        labels=flipped_labels
        return labels

class WitchAttack:
    def __init__(self, class_num):
        self.class_num = class_num

    def attack(self, client_index, labels, mal_idxs):
        # 检查是否是恶意客户端
        if client_index == mal_idxs[0]:
            # 标签翻转：将第 y 类标签翻转为 C-y 类标签
            flipped_labels = torch.tensor([self.class_num - 1 - label.item() for label in labels],
                                           dtype=torch.long).cuda()


        return flipped_labels


import pickle
import numpy as np
from collections import defaultdict
import torch
def sim_matrix(global_hashed_logits):
    # for i in range(40):
    #     for k in range(10):
    #         norm = np.linalg.norm(global_hashed_logits[i][k])
    #         if norm != 0:  # 防止除以0
    #             global_hashed_logits[i][k] /= norm

    similarity_matrix = np.zeros((40, 40))
    # 计算整体的余弦相似度
    for i in range(40):
        for j in range(i, 40):
            if i != j:
                # 计算余弦相似度
                # A = global_hashed_logits[i]  # 第 i 行
                # B = global_hashed_logits[j]  # 第 j 行
                A = global_hashed_logits[i].flatten()
                B = global_hashed_logits[j].flatten()
                # cosine_sim = np.sum(A * B)  # 直接使用点积
                cosine_sim = np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-8)  # 避免除零
                similarity_matrix[i, j] = cosine_sim
                similarity_matrix[j, i] = cosine_sim
    return similarity_matrix

def change_logits(logits):
    # 获取 logits 的行数和列数
    num_rows, num_columns = logits.shape

    # 创建一个空数组用于存放结果
    changed_logits = np.empty_like(logits)

    # 对每一行进行操作
    for i in range(num_rows):
        # 对每一行进行降序排序
        sorted_indices = np.argsort(logits[i])[::-1]  # 获取降序排序的索引
        sorted_values = logits[i][sorted_indices]  # 按照降序索引取值

        # 调整排序顺序，将最大值移到最后
        changed_indices = np.concatenate([sorted_indices[1:], sorted_indices[:1]])
        changed_logits[i, changed_indices] = sorted_values

    return changed_logits


def replace_logits_with_0(log_probs_np):
    # 将输入的 NumPy 数组转换为 PyTorch 张量
    log_probs = torch.from_numpy(log_probs_np)

    # 将所有 logits 设置为0
    log_probs.zero_()

    # 如果需要将结果转换回 NumPy 数组
    return log_probs.numpy()

def replace_logits_with_random(log_probs_np):
    # 将输入的 NumPy 数组转换为 PyTorch 张量
    log_probs = torch.from_numpy(log_probs_np)
    # 指定均值和方差
    mean = 0.0
    std = 1.0
    # 生成具有指定均值和方差的随机数
    random_logits = torch.normal(mean=mean, std=std, size=log_probs.size())
    # 将输入的 logits 替换为随机数
    log_probs.data.copy_(random_logits)
    # 如果需要将结果转换回 NumPy 数组
    return log_probs.numpy()

R_num=16
attack_ind=11

with open('client_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
sim=sim_matrix(loaded_data)
intimacy_table = {i: np.argsort(-sim[i]) for i in range(40)}
# 记录每个客户端的邻居
neighbors_act = {i: intimacy_table[i][:R_num] for i in range(40)}
# print("亲密度列表:{}".format(intimacy_table))
# print("邻居列表:{}".format(neighbors_act))

# 初始化一个字典来存储每个人的出现次数
appearance_count = [0 for _ in range(40)]
# 遍历邻居列表
for array in neighbors_act.values():
    for person in array:
        # 更新出现次数
        appearance_count[person] += 1
print("攻击前出现次数:{}".format(appearance_count))

#攻击
print(neighbors_act)

alpha = 0.5

# 计算每个客户端的权重
weights = {}  # 存储每个客户端的邻居权重
for i in range(40):
    neighbors = neighbors_act[i]
    sim_scores = np.array([sim[i, j] for j in neighbors])  # 获取相似度分数
    # 计算未归一化的权重：w_j = e^(-α(1 - a_ij))
    unnormalized_weights = np.exp(-alpha * (1 - sim_scores))

    # 归一化邻居的权重（不包括客户端本人）
    normalized_weights = unnormalized_weights / np.sum(unnormalized_weights)
    print("邻居节点的权重：{}".format(normalized_weights))


    # 生成客户端i本人的随机权重
    random_weight = np.random.rand()
    print(random_weight)

    # 合并客户端i本人及其邻居的编号
    all_ids = np.append(neighbors, i)
    all_weights = np.append(normalized_weights, random_weight)

    # 根据客户端序号升序排列
    sorted_indices = np.argsort(all_ids)
    sorted_ids = all_ids[sorted_indices]
    sorted_weights = all_weights[sorted_indices]

    # 保存升序排列的权重
    weights[i] = {sorted_ids[j]: sorted_weights[j] for j in range(len(sorted_ids))}

print("权重：{}".format(weights))




# loaded_data[attack_ind]=replace_logits_with_random(loaded_data[attack_ind])
# loaded_data[attack_ind]=change_logits(loaded_data[attack_ind])
loaded_data[attack_ind]=replace_logits_with_0(loaded_data[attack_ind])

sim=sim_matrix(loaded_data)
intimacy_table = {i: np.argsort(-sim[i]) for i in range(40)}
# 记录每个客户端的邻居
neighbors_act = {i: intimacy_table[i][:R_num] for i in range(40)}
# print("亲密度列表:{}".format(intimacy_table))
# print("邻居列表:{}".format(neighbors_act))

print(neighbors_act)

# 初始化一个字典来存储每个人的出现次数
appearance_count = [0 for _ in range(40)]
# 遍历邻居列表

for array in neighbors_act.values():
    for person in array:
        # 更新出现次数
        appearance_count[person] += 1
print("攻击后出现次数:{}".format(appearance_count))


a=[[1,2,3],[4,3,6]]
a.append([[1,2,3],[4,3,6]])
print(a)


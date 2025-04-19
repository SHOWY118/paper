from pypbc import *

# 初始化双线性映射的参数
params = Parameters(qbits=512, rbits=160)
pairing = Pairing(params)

# 生成群G1和G2中的元素g
g = Element.random(pairing, G1)

# 定义向量X和U
X = [1, 2, 3]  # 示例值，可以替换为实际值
U = [4, 5, 6]  # 示例值，可以替换为实际值

# 计算g^X和g^U
X_prime = [g ** x for x in X]
U_prime = [g ** u for u in U]

# 计算∏e(g^x_i, g^u_i)
prod1 = Element.one(pairing, GT)
for x, u in zip(X_prime, U_prime):
    prod1 *= pairing.apply(x, u)

# 计算e(g, g)^(∑x_i * u_i)
sum_xu = sum(x * u for x, u in zip(X, U))
prod2 = pairing.apply(g, g) ** sum_xu

# 输出结果
print("∏e(g^x_i, g^u_i):", prod1)
print("e(g, g)^(∑x_i * u_i):", prod2)
print("验证:", prod1 == prod2)

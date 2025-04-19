from charm.toolbox.pairinggroup import PairingGroup, ZR, G1, pair

# 初始化一个Pairing Group
group = PairingGroup('SS512')  # 使用SS512曲线

# 定义生成元 g
g = group.random(G1)

# 定义X和U向量以及常量sigma
X = [group.random(ZR) for _ in range(3)]  # X=[x1,x2,...,xk]
U = [group.random(ZR) for _ in range(3)]  # U=[u1,u2,...,uk]
sigma = group.random(ZR)  # σ

# 计算左边的乘积
left_product = group.init(G1, 1)
for i in range(len(X)):
    left_product *= pair(g ** X[i], g ** (U[i] + sigma))

# 计算右边的乘积
right_product_1 = group.init(G1, 1)
for i in range(len(X)):
    right_product_1 *= pair(g, g) ** (X[i] * U[i])

right_product_2 = group.init(G1, 1)
for i in range(len(X)):
    right_product_2 *= pair(g, g) ** (X[i] * sigma)

right_product = right_product_1 * right_product_2

# 打印结果
print("Left Product: ", left_product)
print("Right Product: ", right_product)

# 检查是否相等
if left_product == right_product:
    print("The equation holds!")
else:
    print("The equation does not hold.")

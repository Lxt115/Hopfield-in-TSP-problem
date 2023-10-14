'''


连续型——Hopfield神经网络求解TSP
1、初始化权值（A,D,U0）
2、计算N个城市的距离矩阵dxy
3、初始化神经网络的输入电压Uxi和输出电压Vxi
4、利用动力微分方程计算：dUxi/dt
5、由一阶欧拉方法更新计算：Uxi(t+1) = Uxi(t) + dUxi/dt * step
6、由非线性函数sigmoid更新计算：Vxi(t) = 0.5 * (1 + th(Uxi/U0))
7、计算能量函数E
8、检查路径是否合法


'''
import time
import numpy as np
from matplotlib import pyplot as plt


# 代价函数(两城市距离）
# 传入两个城市的点坐标，输出距离
def price_cn(vec1, vec2):
    return pow(pow(vec1[0]-vec2[0], 2)+pow(vec1[1]-vec2[1], 2), 1/2)


def calc_distance(path):
    dis = 0.0
    for i in range(len(path) - 1):
        dis += distance[path[i]][path[i+1]]
    return dis


# 得到城市之间的距离矩阵
# 传入城市点坐标矩阵，输出距离矩阵
def get_distance(citys):
    N = len(citys)
    distance = np.zeros((N, N))  # 初始化距离矩阵
    for i, curr_point in enumerate(citys):
        line = []
        for j, other_point in enumerate(citys):
            if i != j:
                line.append(price_cn(curr_point, other_point))  # 两个不同城市时，将距离加入list中
            else:
                line.append(0.0)  # 自己到自己的距离为0
        distance[i] = line  # 将list加入矩阵
    return distance


# 动态方程计算微分方程du
def calc_du(V, distance):
    a = np.sum(V, axis=0) - 1  # 按列相加
    b = np.sum(V, axis=1) - 1  # 按行相加
    t1 = np.zeros((N, N))  # 初始化t1/t2
    t2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            t1[i, j] = a[j]
    for i in range(N):
        for j in range(N):
            t2[j, i] = b[j]
    # 将第一列移动到最后一列
    c_1 = V[:, 1:N]
    c_0 = np.zeros((N, 1))
    c_0[:, 0] = V[:, 0]
    c = np.concatenate((c_1, c_0), axis=1)
    c = np.dot(distance, c)
    return -A * (t1 + t2) - D * c


# 更新神经网络的输入电压U
def calc_U(U, du, step):
    return U + du * step


# 神经网络的输出电压V更新公式
def calc_V(U, U0):
    return 1 / 2 * (1 + np.tanh(U / U0))


# 计算当前消耗能量
def calc_energy(V, distance):
    v_col = np.sum(V, axis=0)  # 列和
    v_row = np.sum(V, axis=1)  # 行和
    t1 = np.sum(np.power(v_row - 1, 2))
    t2 = np.sum(np.power(v_col - 1, 2))
    idx = []
    for i in range(1, N):
        idx = idx + [i]
    idx = idx + [0]
    Vt = V[:, idx]
    t3 = distance * Vt
    # t3表示当前路径的总长度
    t3 = np.sum(np.sum(np.multiply(V, t3)))  # multiply矩阵点乘
    e = 0.5 * (A * (t1 + t2) + D * t3)
    return e


# 检查路径的正确性
# 满足条件则认为是一个次优解，若能量小于最小能量则记录其能量
def check_path(V):
    newV = np.zeros([N, N])
    route = []
    for i in range(N):
        mm = np.max(V[:, i])
        for j in range(N):
            if V[j, i] == mm:
                newV[j, i] = 1
                route += [j]
                break
    return route, newV


# 可视化回路图和能量变化趋势（迭代次数-能量值）
def draw_H_and_E(citys, H_path, energys):
    fig = plt.figure()

    # 绘制哈密顿回路
    ax1 = fig.add_subplot(121)  # 画子图

    for (from_, to_) in H_path:
        # 绘制城市点
        p1 = plt.Circle(citys[from_], 0.02, color='green')
        p2 = plt.Circle(citys[to_], 0.02, color='green')
        ax1.add_patch(p1)
        ax1.add_patch(p2)
        # 绘制路线
        ax1.plot((citys[from_][0], citys[to_][0]), (citys[from_][1], citys[to_][1]), color='blue')
        # 打印城市名（1、2、3……）
        ax1.annotate(text=to_+1, xy=citys[to_], xytext=(-7, -8), textcoords='offset points', fontsize=8)
    ax1.axis('equal')
    # 绘制网格
    ax1.grid()

    # 绘制能量趋势图
    ax2 = fig.add_subplot(122)
    ax2.plot(np.arange(0, len(energys), 1), energys, color='blue')
    ax2.grid()
    plt.show()


if __name__ == '__main__':
    # 随机生成城市坐标点矩阵
    x = np.random.rand(47)
    y = np.random.rand(47)
    cities = []
    for i in range(47):
        cities = cities + [[x[i], y[i]]]
    print(cities)
    """
    cities = []
    with open("城市坐标.txt", "r", encoding="UTF-8") as f:
        line = f.readline().strip()
        while line:
            l = line.split(",")
            l[0] = int(l[0])
            l[1] = int(l[1])
            cities.append(l)
            line = f.readline().strip()
    """
    N = len(cities)  # N为城市个数
    start = time.time()
    # 得到城市之间的距离矩阵
    distance = get_distance(cities)

    # 设置惩罚因子值
    A = 500
    D = 200

    U0 = 0.0009  # 初始电压
    step = 0.0001  # 步长
    num_iter = 10000  # 迭代次数

    # 初始化神经网络的输入状态（电路的输入电压U）
    U = 1 / 2 * U0 * np.log(N - 1) + (20 * (np.random.random((N, N))) - 1)  # 加上噪声
    # 初始化神经网络的输出状态（电路的输出电压V）
    V = calc_V(U, U0)
    energy = np.array([0.0 for x in range(num_iter)])  # 每次迭代的能量
    best_distance = np.inf  # 最优距离
    best_route = []  # 最优路线
    H_path = []  # 哈密顿回路

    # 开始迭代训练网络
    for n in range(num_iter):
        # 利用动态方程计算du
        du = calc_du(V, distance)
        # 由一阶欧拉法更新下一个时间的输入状态（电路的输入电压U）
        U = calc_U(U, du, step)
        # print(U)
        # 由sigmoid函数更新下一个时间的输出状态（电路的输出电压V）
        V = calc_V(U, U0)
        # 计算当前网络的能量E
        energy[n] = calc_energy(V, distance)
        # 检查路径的合法性
        route, newV = check_path(V)
        # print(route)
        # print(np.unique(route))
        if len(np.unique(route)) == N:
            route.append(route[0])
            dis = calc_distance(route)
            # print(dis)
            # print(best_distance)

            if dis < best_distance:
                H_path = []
                best_distance = dis
                best_route = route
                for i in range(len(route) - 1):
                    H_path.append((route[i], route[i + 1]))
                print('第{}次迭代找到的次优解距离为：{}，能量为：{}，路径为：'.format(n, best_distance, energy[n]))
                for i, v in enumerate(best_route):
                    if i < len(best_route) - 1:
                        print(v+1, end='->')
                    else:
                        print(v+1, end='\n')
    end = time.time()
    print('运行时间：%s 秒'%(end - start))
    if len(H_path) > 0:
        draw_H_and_E(cities, H_path, energy)
    else:
        print('没有找到最优解')

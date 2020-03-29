# -*- coding: utf-8 -*-
# @Time    : 20/03/24 21:00
# @Author  : Wang Yu
# @Project : 09 Statistical computing
# @File    : one.py
# @Software: PyCharm


import time

import numpy as np

N = 1000  # 数据维度
D = 10  # 数据维度
IT_MAX = 5000  # 最大迭代次数
EPS = 1e-4  # 允许误差
SEED = 100  # 设置随机点
STEP_S = 'Armijo'  # "None"-不使用搜索策略恒定0.01学习率；"Armijo"-使用Armijo非精确步长搜索法
DATA_TYPE = 0


def raw_data(flag_print=False, s=SEED):
    """    生成题设原始数据
    :param flag_print: (bool)打印开关
    :param s: (int)设置随机种子
    :return X: (np.array((N, D)))
    :return Y: (np.array(N))
    :return beta_true: (np.array(D))
    """
    np.random.seed(s)  # set the random seed
    X = np.random.normal(size=(N, D))
    beta_true = np.random.normal(size=D)
    Y = 1 / (1 + np.exp(-X.dot(beta_true)))
    Y = np.zeros(N)
    Y[1 / (1 + np.exp(-X.dot(beta_true))) > 0.5] = 1
    if flag_print:
        print("raw data:")
        print(f"X\t\te.g:{X[0][:3]}...\tshape:{X.shape}")
        print(f"Y\t\te.g:{Y[:3]}...\tshape:{Y.shape}")
        print(f"b_true\te.g:{beta_true[:3]}...\tshape:{beta_true.shape}")
    return X, Y, beta_true


def sparse_data(flag_print=False, s=SEED, sparse_rate=0.3):
    """    生成题设稀疏数据
    :param flag_print: (bool)打印开关
    :param s: (int)设置随机种子
    :param sparse_rate: (float)稀疏率
    :return X: (np.array((N, D)))
    :return Y: (np.array(N))
    :return beta_true: (np.array(D))
    """
    np.random.seed(s)  # set the random seed
    X = np.random.normal(size=(N, D))
    M = np.random.uniform(size=(N, D)) < sparse_rate
    X[M] = 0
    beta_true = np.random.normal(size=D)
    Y = 1 / (1 + np.exp(-X.dot(beta_true)))
    if flag_print:
        print("sparse data:")
        print(f"X\t\te.g:{X[0][:3]}...\tshape:{X.shape}")
        print(f"Y\t\te.g:{Y[:3]}...\tshape:{Y.shape}")
        print(f"b_true\te.g:{beta_true[:3]}...\tshape:{beta_true.shape}")
    return X, Y, beta_true


def load_data(flag_print=False):
    """    加载适用题目的数据
    :return: 原始数据或稀疏数据
    """
    if DATA_TYPE == 0:
        return raw_data(flag_print)
    elif DATA_TYPE == 1:
        return sparse_data(flag_print)
    else:
        print("Data selection error!")


def obj_func(beta):
    """    本题的目标函数
    :param beta: (np.array(D))迭代beta值
    :return: (float)目标函数值
    """
    X, Y, _ = load_data()
    L = Y.dot(X.dot(beta)) - np.log(1 + np.exp(X.dot(beta))).sum()
    return -L


def grad_func(beta):
    """    目标函数的梯度
    :param beta: (np.array(D))迭代beta值
    :return: (np.array(D)目标函数梯度值
    """
    X, Y, _ = load_data()
    g_L = np.zeros(len(beta))
    for i in range(len(Y)):
        g_L += Y[i] * X[i] - X[i] / (1 + np.exp(-X[i].dot(beta)))
    return -g_L


def Hessian(beta):
    """    求取目标函数的海森阵
    :param beta: : (np.array(D))迭代beta值
    :return: (np.array((D, D)))目标函数海森阵值
    """
    He = np.zeros([len(beta), len(beta)])
    X, Y, _ = load_data()
    for i in range(len(Y)):
        He += (-np.exp(-X[i].dot(beta)) / (1 + np.exp(-X[i].dot(beta))) ** 2) * (
            X[i].reshape(-1, 1).dot(X[i].reshape(1, -1)))
    return He


def Armijo(x_k, d_k, beta=0.55, sigma=0.4):
    """    Armijo非精确步长搜索算法
    :param x_k: (array)迭代点
    :param d_k: (array)当前迭代点处方向
    :param beta: (float)算法参数
    :param sigma: (float)算法参数
    :return: (float)步长计算结果
    """
    for m in range(20):
        if obj_func(x_k + beta ** m * d_k) < obj_func(x_k) + sigma * (beta ** m) * (grad_func(x_k)).dot(d_k):
            break
    return beta ** m


def step_search(x_k, d_k):
    """    确定步长搜索方法，固定值或是非精确搜索
    :param x_k: (array)迭代点
    :param d_k: (array)当前迭代点处方向
    :return: (float)步长计算结果
    """
    if STEP_S == "None":
        return 0.01
    if STEP_S == "Armijo":
        return Armijo(x_k, d_k)
    else:
        print("Step search methods selection error!")


def gradient_descent(x_0, it_max=IT_MAX, eps=EPS):
    """    通用梯度下降法
    :param x_0: (array)初始值
    :param it_max: (int)最大迭代次数
    :param eps: (float)停止条件
    :return result: (list of array)迭代结果列表
    :return : (float)运行时间
    """
    start = time.time()
    result = [x_0]
    for i in range(it_max):
        d_k = -grad_func(result[-1])
        alpha_k = step_search(result[-1], d_k)
        x_k = result[-1] + alpha_k * d_k
        if (np.linalg.norm(d_k))**2 < eps:
            break
        result.append(x_k)
        print(f"GD\t  it:{i}\t  {obj_func(result[-1]):0.2f}")

    return np.array(result), time.time() - start


def Newton_method(x_0, it_max=IT_MAX, eps=EPS):
    """    通用牛顿法
    :param x_0: (array)初始值
    :param it_max: (int)最大迭代次数
    :param eps: (float)停止条件
    :return result: (list of array)迭代结果列表
    :return : (float)运行时间
    """
    start = time.time()
    result = [x_0]
    for i in range(it_max):
        g_k = grad_func(result[-1])
        d_k = np.linalg.solve(Hessian(result[-1]), g_k).reshape(-1)
        alpha_k = step_search(result[-1], d_k)
        x_k = result[-1] + alpha_k * d_k
        if (np.linalg.norm(d_k))**2 < eps:
            break
        result.append(x_k)
        print(f"Newton\t  it:{i}\t  {obj_func(result[-1]):0.2f}")
    return np.array(result), time.time() - start


def Quasi_Newton_R1(x_0, it_max=IT_MAX, eps=EPS):
    """
    通用拟牛顿法对称秩1更新
    :param x_0: (array)初始值
    :param it_max: (int)最大迭代次数
    :param eps: (float)停止条件
    :return result: (list of array)迭代结果列表
    :return : (float)运行时间
    """
    start = time.time()
    B_k = np.identity(len(x_0))
    result = [x_0]
    for i in range(it_max):
        g_k = grad_func(result[-1])
        d_k = -B_k @ g_k
        alpha_k = step_search(result[-1], d_k)
        x_k = result[-1] + alpha_k * d_k

        s_k = x_k - result[-1]
        y_k = grad_func(x_k) - g_k
        a_k = s_k - B_k @ y_k
        B_k = B_k + a_k.reshape(-1, 1).dot(a_k.reshape(1, -1)) / (np.dot(a_k, y_k))
        if (np.linalg.norm(d_k))**2 < eps:
            break
        result.append(x_k)
        print(f"QN_R1\t  it:{i}\t  {obj_func(result[-1]):0.2f}")

    return np.array(result), time.time() - start


def Quasi_Newton_BFGS(x_0, it_max=IT_MAX, eps=EPS):
    """
    通用拟牛顿法BFGS更新
    :param x_0: (array)初始值
    :param it_max: (int)最大迭代次数
    :param eps: (float)停止条件
    :return result: (list of array)迭代结果列表
    :return : (float)运行时间
    """
    start = time.time()
    B_k = np.identity(len(x_0))
    result = [x_0]
    for i in range(it_max):
        g_k = grad_func(result[-1])
        d_k = np.linalg.solve(-B_k, g_k)
        alpha_k = step_search(result[-1], d_k)
        x_k = result[-1] + alpha_k * d_k
        s_k = x_k - result[-1]
        y_k = grad_func(x_k) - g_k
        if y_k.dot(s_k) > 0:
            s_k = s_k.reshape(-1, 1)
            y_k = y_k.reshape(-1, 1)
            B_k = B_k - (B_k @ s_k @ s_k.T @ B_k) / (s_k.T @ B_k @ s_k) + (y_k @ y_k.T) / (y_k.T @ s_k)
        if (np.linalg.norm(d_k))**2 < eps:
            break
        result.append(x_k)
        print(f"QN_BFGS\t  it:{i}\t  {obj_func(result[-1]):0.2f}")

    return np.array(result), time.time() - start


def Fisher_scoring(x_0, it_max=IT_MAX, eps=EPS):
    """
    非通用Fisher得分优化算法
    :param x_0: (array)初始值
    :param it_max: (int)最大迭代次数
    :param eps: (float)停止条件
    :return result: (list of array)迭代结果列表
    :return : (float)运行时间
    """
    start = time.time()
    result = [x_0]
    for i in range(it_max):
        g_k = grad_func(result[-1])
        d_k = np.linalg.solve(Hessian(result[-1]), g_k).reshape(-1)
        alpha_k = step_search(result[-1], d_k)
        x_k = result[-1] + alpha_k * d_k
        if (np.linalg.norm(x_k - result[-1]))**2 < eps:
            break
        result.append(x_k)
        print(f"fisher\t  it:{i}\t  {obj_func(result[-1]):0.2f}")

    return np.array(result), time.time() - start


def test1():
    print(f"N = {N}, D = {D}")
    w_0 = np.zeros(D)  # 初值选取

    """1 梯度下降法"""
    print("梯度下降法：")
    start = time.time()
    w, error = gradient_descent(w_0)
    #    print(w=w_0)
    print(f"耗时{time.time() - start:0.2f}s")
    print(w)

    """2 牛顿法"""
    print("牛顿法：")
    start = time.time()
    w, error = Newton_method(w_0)
    #    print(w=w_0)
    print(f"耗时{time.time() - start:0.2f}s")
    print(w)
    """3 拟牛顿法R1"""
    print("拟牛顿法R1:")
    start = time.time()
    w, error = Quasi_Newton_R1(w_0)
    #    print(w=w_0)
    print(f"耗时{time.time() - start:0.2f}s")
    print(w)
    """4 拟牛顿法BFGS"""
    print("拟牛顿法BFGS:")
    start = time.time()
    w = Quasi_Newton_BFGS(w_0)
    #    print(w=w_0)
    print(f"耗时{time.time() - start:0.2f}s，")
    print(w)


def test2():
    beta_0 = np.zeros(D)
    raw_data(1)
    """1 梯度下降法"""
    print("梯度下降法：")
    start = time.time()
    result1 = gradient_descent(beta_0)
    print(f"迭代次数：{len(result1)}\t耗时{time.time() - start:0.2f}s")
    print(f"结果：{result1[-1][:3]}")
    """2 牛顿法"""
    print("牛顿法：")
    start = time.time()
    result2 = Newton_method(beta_0)
    print(f"迭代次数：{len(result2)}\t耗时{time.time() - start:0.2f}s")
    print(f"结果：{result2[-1][:3]}")

    """3 拟牛顿法R1"""
    print("拟牛顿法R1：")
    start = time.time()
    result3 = Quasi_Newton_R1(beta_0)
    print(f"迭代次数：{len(result3)}\t耗时{time.time() - start:0.2f}s")
    print(f"结果：{result3[-1][:3]}")

    """4 拟牛顿法BFGS"""
    print("拟牛顿法R1：")
    start = time.time()
    result4 = Quasi_Newton_BFGS(beta_0)
    print(f"迭代次数：{len(result4)}\t耗时{time.time() - start:0.2f}s")
    print(f"结果：{result4[-1][:3]}")

    """5 Fisher得分法"""
    print("Fisher得分法：")
    start = time.time()
    result5 = Fisher_scoring(beta_0)
    print(f"迭代次数：{len(result5)}\t耗时{time.time() - start:0.2f}s")
    print(f"结果：{result5[-1][:3]}")


def main():
    X, Y, b_true = load_data(flag_print=0)
    b0 = np.zeros(D)
    print("Start：")
    b1, t1 = gradient_descent(b0)
    b2, t2 = Newton_method(b0)
    b3, t3 = Quasi_Newton_R1(b0)
    b4, t4 = Quasi_Newton_BFGS(b0)
    b5, t5 = Fisher_scoring(b0)
    print(f"beta_true\t{b_true.shape} {b_true[:2]}")
    print(f"GD \t b1{b1[-1][:2]}\t{t1:3.2f}s \t it:{len(b1)}")
    print(f"Newton \t b2{b2[-1][:2]}\t{t2:3.2f}s \t it:{len(b2)}")
    print(f"QN_R1 \t b3{b3[-1][:2]}\t{t3:3.2f}s \t it:{len(b3)}")
    print(f"BFGS \t b4{b4[-1][:2]}\t{t4:3.2f}s \t it:{len(b4)}")
    print(f"Fisher \t b5{b5[-1][:2]}\t{t5:3.2f}s \t it:{len(b5)}")
    b_result = np.array([b1,b2,b3,b4,b5])
    t_result = np.array([t1,t2,t3,t4,t5])
    return b_result, t_result, 

def painting():
    B,T = main()
    ERR_n = []
    ERR_t = []
    for i in range(len(B)):
        err = np.linalg.norm(B[i]-B[i][-1:],axis=1)
        err = err/err[0]
        x_old = range(len(err))
        x_new = np.linspace(0,len(err),int(10*T[i]))
        y = np.interp(x_new, x_old, err)
        ERR_n.append(list(err))
        ERR_t.append(list(y))
        
        
        
        


if __name__ == '__main__':
    B,T = main()
    _, _, b_true = load_data(flag_print=0)
    ERR_n = []
    ERR_t = []
    for i in range(len(B)):
        err = np.linalg.norm(B[i]-b_true,axis=1)
        err = err/err[0]
        x_old = range(len(err))
        x_new = np.linspace(0,len(err),int(10*T[i]))
        y = np.interp(x_new, x_old, err)
        ERR_n.append(list(err))
        ERR_t.append(list(y))

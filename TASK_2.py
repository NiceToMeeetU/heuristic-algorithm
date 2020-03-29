# -*- coding: utf-8 -*-
# @Time    : 20/03/25 9:28
# @Author  : Wang Yu
# @Project : 09 Statistical computing
# @File    : TASK_2.py
# @Software: PyCharm


import math
import random
import time

import numpy as np
from sklearn.cluster import KMeans


def load_data():
    """    加载数据    """
    data = np.loadtxt("data.dat", delimiter=' ', skiprows=1)
    return data[:, 1:]


def load_initial_label():
    """    加载原始标签    """
    data = np.loadtxt("data.dat", delimiter=' ', skiprows=1)
    return data[:, 0]


def object_func(label_in):
    """    本题目标函数
    :param label_in: (np.array)标签数组{0,1,2}
    :return var: (float)组内平方和之和
    """
    var = 0
    data = load_data()
    for i in set(label_in):
        data_in_label = data[label_in == i]
        var += sum(sum((data_in_label - np.mean(data_in_label, axis=0)) ** 2))
    return var


def object_funcs(labels_in):
    """    批量计算目标函数
    :param labels_in: (list of array)标签数据列表
    :return result: (list of float)组内平方和列表
    """
    result = []
    data = load_data()
    for one in labels_in:
        var = 0
        for i in set(one):
            data_in_label = data[one == i]
            var += sum(sum((data_in_label - np.mean(data_in_label, axis=0)) ** 2))
        result.append(var)
    return np.array(result)


def Simulated_Annealing_old(c_0, alpha=0.9, TEMP=100.0, LOW_TEMP=0.1, iter_in=50):
    """    通用模拟退火算法
    :param c_0: (np.array)标签数组初值
    :param alpha: (float)退火率
    :param TEMP: (float)初始温度
    :param LOW_TEMP:(float)最低温度
    :param iter_in:(int)内层迭代次数
    :return c_result: (list of np.array)迭代标签数组列表
    :return f_result: (np.array)迭代结果数组
    """

    def switch(c_in):
        """
        随机切换标签内的任意两值
        :param c_in: (np.array)标签数组{0,1,2}
        :return label: (np.array)更新后的标签数组
        """
        c_out = c_in.copy()  # 深浅拷贝，务必注意，极易出错！！
        i_ = random.randint(0, len(c_out) - 1)
        labels = list(set(c_out))
        labels.remove(c_out[i_])
        c_out[i_] = random.choice(labels)
        return c_out

    start = time.time()
    tmp = TEMP  # 设定退火初始温度
    lowest_tmp = LOW_TEMP  # 设定最低退火温度
    c_result = [c_0]  # 将初值标签push入结果列表
    f_result = [object_func(c_0)]  # 将初值结果push入结果列表
    i = 0
    while tmp > lowest_tmp:  # 外层降温循环
        # 构建内层循环临时数组，用于得到外层迭代初始的最优结果
        c_temp = [c_result[-1]]
        f_temp = [object_func(c_temp[-1])]
        for _ in range(iter_in):
            c1 = switch(c_temp[-1])  # 交换数组内元素
            f1 = object_func(c1)
            delta = f1 - f_temp[-1]
            if delta < 0:
                c_temp.append(c1)
                f_temp.append(f1)
            elif math.exp(-delta / tmp) > random.random():
                c_temp.append(c1)
                f_temp.append(f1)

        min_temp = np.argmin(f_temp)
        c_result.append(c_temp[min_temp])
        f_result.append(f_temp[min_temp])
        tmp = tmp * alpha  # 退火
        i += 1
        print(f"SA  it:{i}  {f_result[-1]}")
        if i > 20:
            out_it = f_result[-20:]
            if not (out_it - out_it[0]).any() and f_result[-1] == 1270.9388375049873:
                break
    return np.array(c_result), np.array(f_result), time.time() - start


def Simulated_Annealing(c_0, alpha=0.8, TEMP=900.0, LOW_TEMP=0.01, iter_in=100):
    """    通用模拟退火算法
    :param c_0: (np.array)标签数组初值
    :param alpha: (float)退火率
    :param TEMP: (float)初始温度
    :param LOW_TEMP:(float)最低温度
    :param iter_in:(int)内层迭代次数
    :return c_result: (list of np.array)迭代标签数组列表
    :return f_result: (np.array)迭代结果数组
    """

    def switch(c_in):
        """
        随机切换标签内的任意两值
        :param c_in: (np.array)标签数组{0,1,2}
        :return label: (np.array)更新后的标签数组
        """
        c_out = c_in.copy()  # 深浅拷贝，务必注意，极易出错！！
        i_ = random.randint(0, len(c_out) - 1)
        labels = list(set(c_out))
        labels.remove(c_out[i_])
        c_out[i_] = random.choice(labels)
        return c_out

    start = time.time()
    tmp = TEMP  # 设定退火初始温度
    lowest_tmp = LOW_TEMP  # 设定最低退火温度
    c_result = [c_0]  # 将初值标签push入结果列表
    f_result = [object_func(c_0)]  # 将初值结果push入结果列表
    i = 0
    while tmp > lowest_tmp:  # 外层降温循环
        # 构建内层循环临时数组，用于得到外层迭代初始的最优结果
        c_temp = [c_result[-1]]
        f_temp = [object_func(c_temp[-1])]
        j = 0
        for _ in range(iter_in):
            c1 = switch(c_temp[-1])  # 交换数组内元素
            f1 = object_func(c1)
            delta = f1 - f_temp[-1]
            if delta < 0:
                c_temp.append(c1)
                f_temp.append(f1)
                j += 1
            elif math.exp(-delta / tmp) > random.random():
                c_temp.append(c1)
                f_temp.append(f1)
                j += 1

            if j > 0.5 * iter_in:
                break
        min_temp = np.argmin(f_temp)
        c_result.append(c_temp[min_temp])
        f_result.append(f_temp[min_temp])
        tmp = tmp * alpha  # 退火
        i += 1
        print(f"SA  it:{i}  {f_result[-1]}")
        if i > 20:
            out_it = f_result[-20:]
            if not (out_it - out_it[0]).any() and f_result[-1] == 1270.9388375049873:
                break
    return np.array(c_result), np.array(f_result), time.time() - start


def Genetic_algorithm(c_0, pop_size=100, cross_rate=0.8, mutation_rate=0.0075, n_gen=500):
    """
    非通用遗传算法
    :param c_0: (np.array)标签数组初值
    :param pop_size: (int)种群规模
    :param cross_rate: (float)交叉概率
    :param mutation_rate: (float)变异概率
    :param n_gen: (int)遗传代数
    :return c_result: (list of np.array)迭代标签数组列表
    :return f_result: (np.array)迭代结果数组
    """
    start = time.time()
    DNA_size = len(c_0)  # DNA长度

    def get_fitness(pred):  # 适应性，本题函数值越小越适应
        return np.exp(np.max(pred) + 1e-3 - pred)

    def select(pop_in, fitness_in):  # 自然选择过程
        idx = np.random.choice(np.arange(pop_size), size=pop_size, replace=True,
                               p=fitness_in / fitness_in.sum())
        return pop_in[idx]

    # 有限标签型优化问题自带DNA效果，不用编码解码
    # def translateDNA(pop_in):
    #     return pop_in

    def crossover(parent_in, pop_in):  # 交配过程
        if np.random.rand() < cross_rate:
            i_ = np.random.randint(0, pop_size, size=1)
            cross_points = np.random.randint(0, 2, size=DNA_size).astype(np.bool)
            parent_in[cross_points] = pop_in[i_, cross_points]
        return parent_in

    def mutate(child_in):  # 变异过程
        child_out = child_in.copy()
        for child_index in range(DNA_size):
            if np.random.rand() < mutation_rate:
                labels = [1, 2, 3]
                labels.remove(child_in[child_index])
                child_out[child_index] = random.choice(labels)
        return child_out

    c_result = [c_0]  # 将初值标签push入结果列表
    f_result = [object_func(c_0)]  # 将函数值push入结果列表
    pop = np.random.randint(1, 4, size=(pop_size, DNA_size))  # 初始化DNA

    for i in range(n_gen):
        F_values = object_funcs(pop)  # 批量计算DNA对应函数值
        fitness = get_fitness(F_values)  # 计算适应性
        c_result.append(pop[np.argmax(fitness)])
        f_result.append(F_values[np.argmax(fitness)])  # 存入最好觉果
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child  # 子代替换父代
        print(f"GA  it:{i}  {f_result[-1]}")
        if i > 20:
            out_it = f_result[-20:]
            if not (out_it - out_it[0]).any() and f_result[-1] == 1270.9388375049873:
                break

    return np.array(c_result), np.array(f_result), time.time() - start


def Tabu_search(c_0, tabu_len=10, it_max=150):
    """
        非通用禁忌搜索
        :param c_0: (np.array)标签数组初值
        :param tabu_len: (int)禁忌表长度
        :param it_max: (int)最大迭代次数
        :return c_result: (list of np.array)迭代标签数组列表
        :return f_result: (np.array)迭代结果数组
        """
    start = time.time()

    def find_near(c_in):  # 查找近邻解
        c_near = []
        for i_ in range(len(c_in)):
            labels = [1, 2, 3]
            labels.remove(c_in[i_])
            for j_ in range(len(labels)):
                c_new = c_in.copy()
                c_new[i_] = random.choice(labels)
                c_near.append(c_new)
        return np.array(c_near)

    c_result = [c_0]  # 将初值标签push入结果列表
    f_result = [object_func(c_0)]
    tabu_list = []
    c_best = c_result[-1].copy()
    for i in range(it_max):
        cs_new = find_near(c_best)
        fs_new = object_funcs(cs_new)
        fs_index = np.argsort(fs_new)
        for j in fs_index:
            if j not in tabu_list:
                c_best = cs_new[j].copy()
                c_result.append(cs_new[j])
                f_result.append(fs_new[j])
                tabu_list.append(j)
                break
            elif fs_new[j] < np.min(f_result):
                c_best = cs_new[j].copy()
                c_result.append(cs_new[j])
                f_result.append(fs_new[j])
                tabu_list.remove(j)
                tabu_list.append(j)
                break

        if len(tabu_list) > tabu_len:
            tabu_list.pop(0)
        print(f"TS  it:{i}  {f_result[-1]}")

        if i > 20:
            out_it = f_result[-20:]
            if not (out_it - out_it[0]).any() and f_result[-1] == 1270.9388375049873:
                break
    f_result[-1] = np.min(f_result)
    c_result[-1] = c_result[np.argmin(f_result)]
    return np.array(c_result), np.array(f_result), time.time() - start


def K_means():
    """    聚类验证
    """
    data = load_data()
    estimator = KMeans(3)
    estimator.fit(data)
    c_result = estimator.labels_
    f_result = estimator.inertia_
    return c_result, f_result


def main():
    data = load_data()
    c0 = np.random.randint(1, 4, len(data))  # 优化初值
    print("开始计算...")
    c1, f1, t1 = Simulated_Annealing(c0, 0.8, 800, 0.01, 100)
    c2, f2, t2 = Genetic_algorithm(c0, 100, 0.8, 0.0075, 500)
    c3, f3, t3 = Tabu_search(c0, 10, 150)
    print("计算完成...")
    fu = [list(f1), list(f2), list(f3)]
    tu = [t1, t2, t3]
    print(f"SA 耗时{t1:0.2f}s 迭代{len(f1)}次 终值{f1[-1]:0.3f}")
    print(f"GA 耗时{t2:0.2f}s 迭代{len(f2)}次 终值{f2[-1]:0.3f}")
    print(f"LS 耗时{t3:0.2f}s 迭代{len(f3)}次 终值{f3[-1]:0.3f}")
    return fu, tu


def painting():
    pass
    fu, tu = main()
    tt = [int(100 * i) for i in tu]
    xx = [range(len(i)) for i in fu]
    pp = [np.interp(np.linspace(0, len(fu[i]), tt[i]), xx[i], fu[i]) for i in range(3)]
    pp = [list(i) for i in pp]
    return pp


def best_parameter1():
    print("SA调参")
    data = load_data()
    c0 = np.random.randint(1, 4, len(data))  # 优化初值
    """
    :param alpha: (float)退火率，默认取0.9
    :param TEMP: (float)初始温度
    :param LOW_TEMP:(float)最低温度
    :param iter_in:(int)内层迭代次数"""
    result_p1 = []
    i = 1
    for alpha in [0.6, 0.7, 0.8, 0.9]:
        for temp in [100, 300, 500, 700, 900]:
            for low_temp in [0.01, 0.1, 1]:
                for iter_in in [50, 75, 100, 125, 150, 180]:
                    c1, f1, t1 = Simulated_Annealing(c0, alpha, temp, low_temp, iter_in)
                    result_p1.append(np.array([i, alpha, temp, low_temp, iter_in, len(f1), t1, f1[-1]]))
                    print(f"SA1  {i}   {t1:0.2f}s  {len(f1)}   {f1[-1]:0.3f}")
                    i += 1
    return np.array(result_p1)


def best_parameter2():
    print("GA调参")
    data = load_data()
    c0 = np.random.randint(1, 4, len(data))  # 优化初值
    """
    :param pop_size: (int)种群规模
    :param cross_rate: (float)交叉概率
    :param mutation_rate: (float)变异概率
    :param n_gen: (int)遗传代数
    """
    result_p2 = []
    i = 1
    for pop_size in [100, 200, 300, 400, 500]:
        for cross_rate in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for mutation_rate in [0.01, 0.0075, 0.005, 0.0025]:
                for n_gen in [100, 300, 500, 700]:
                    c2, f2, t2 = Genetic_algorithm(c0, pop_size, cross_rate, mutation_rate, n_gen)
                    result_p2.append(np.array([i, pop_size, cross_rate, mutation_rate, n_gen, len(f2), t2, f2[-1]]))
                    print(f"SA  {i}   {t2:0.2f}s  {len(f2)}   {f2[-1]:0.3f}")
                    i += 1
    return np.array(result_p2)


def best_parameter3():
    print("GA调参")
    data = load_data()
    c0 = np.random.randint(1, 4, len(data))  # 优化初值
    """
    :param tabu_len: (int)禁忌表长度
    :param it_max: (int)最大迭代次数
    """
    result_p3 = []
    i = 1
    for tabu_len in [10, 20, 30, 40, 50]:
        for it_max in [100, 200, 300, 400, 500, 600]:
            c3, f3, t3 = Tabu_search(c0, tabu_len, it_max)
            result_p3.append(np.array([i, tabu_len, it_max, len(f3), t3, f3[-1]]))
            print(f"SA  {i}   {t3:0.2f}s  {len(f3)}   {f3[-1]:0.3f}")
            i += 1
    return np.array(result_p3)


if __name__ == '__main__':
    main()

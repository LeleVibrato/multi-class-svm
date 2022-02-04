# -*- coding: utf-8 -*-
# @时间 : 2022年2月3日
# @作者 : sam255
# @文件名 : nlsvm.py
# @软件 : SVM多分类器
# @协议 : MIT

import numpy as np
from scipy import sparse
import osqp
from functools import partial
from typing import Callable, Literal


def nlsvm_solve(X: np.ndarray, y: np.ndarray, classes: tuple[str, str], cost: float, kernel: Callable) -> Callable:
    """
    训练一个以多项式为核函数的SVM二分类器
    :param X: 训练数据的特征矩阵，n行，dim列，dim为数据维数，每一行为一个训练样本，每一列为一个特征
    :param y: 训练数据的标签矩阵，n行，1列，取值为-1或1
    :param classes: 长度为2的字符串元组，表示当前用于训练的二分类
    :param cost: SVM超参数，alpha_i的上界
    :param kernel: 多项式核函数的类型
    :return closure: SVM二分类器
    """
    n = X.shape[0]  # 样本个数

    # 计算内积
    inprod = np.ones((n, n))
    for i in range(n):
        for j in range(i+1):
            inprod[i, j] = inprod[j, i] = kernel(X[j, :], X[i, :])

    # 定义二次规划参数矩阵
    q = -np.ones(n)
    P = 0.5 * y @ y.T * inprod
    P[np.diag_indices_from(P)] = 2 * P[np.diag_indices_from(P)]
    A = np.vstack((np.diag(np.ones(n)), y.T))
    lb = np.zeros(n+1)
    ub = cost * np.ones(n+1)
    ub[-1] = 0.0

    # 求解二次规划
    prob = osqp.OSQP()
    prob.setup(sparse.csc_matrix(P), q,
               sparse.csc_matrix(A), lb, ub, verbose=False)
    res = prob.solve()
    alpha = res.x

    # 寻找支持向量的索引值
    svs_ind = np.where(np.logical_and(0.0 < alpha, alpha < cost))[0]

    # 计算 beta0（取平均值）
    beta0sum = 0.0
    for j in svs_ind:
        beta0sum += 1.0 / y[j, 0] - sum(alpha[i] * y[i, 0] * inprod[j, i]
                                        for i in range(n))
    beta0 = beta0sum / len(svs_ind)  # 取平均值

    def nlsvm_classifer(x: np.ndarray) -> str:
        """
        SVM 二分类器，作为函数闭包被上一级函数返回
        :param x: 1维np.array，长度为dim
        :return str: 样本 x 的预测类别
        """
        result = beta0 + \
            sum(alpha[i] * y[i, 0] * kernel(x, X[i, :])
                for i in range(n))
        if result >= 0.0:
            return classes[0]
        else:
            return classes[1]

    return nlsvm_classifer


def ovo(y: np.ndarray) -> list[tuple[str, str]]:
    """
    将类别拆分成多个一对一元组
    :param y: 一维的字符串列表，代表各个训练样本的分类
    :return : 各个分类器要进行的二分类类别
    :rtype list: 二元元组列表，元组元素类型为字符串
    """
    unique_class = np.unique(y)
    unique_class.sort()
    num_of_classes = len(unique_class)  # 类别数
    print("类别: ", unique_class)
    return [(unique_class[i], unique_class[j])
            for i in range(num_of_classes) for j in range(i+1, num_of_classes)]


def get_kernel(kernel: str, kargs: dict) -> Callable:
    """
    根据参数生成核函数
    :param kernel: 核函数类型
    :param kargs: 核函数超参数
    :return Callable: 核函数 
    """
    if kernel == "poly":
        def poly_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            """
            多项式核函数
            :param x1: 第一个向量
            :param x2: 第二个向量
            :return float: 两个向量的内积
            """
            return (1+np.dot(x1, x2))**kargs["degree"]
        return poly_kernel

    elif kernel == "linear":
        def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            """
            线性核函数
            :param x1: 第一个向量
            :param x2: 第二个向量
            :return float: 两个向量的内积
            """
            return np.dot(x1, x2)
        return linear_kernel

    elif kernel == "rbf":
        def gaussian_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
            """
            高斯核函数
            :param x1: 第一个向量
            :param x2: 第二个向量
            :return float: 两个向量的内积
            """
            return np.exp(-np.linalg.norm(x1-x2)**2 / (2 * (kargs["sigma"] ** 2)))
        return gaussian_kernel

    else:
        raise ValueError("不支持的核函数")


def nlsvm(X: np.ndarray, y: np.ndarray, cost:  float, kernel: Literal["linear", "poly", "rbf"], **kargs) -> list[Callable]:
    """
    模型训练的接口函数
    :param X: 训练数据的特征矩阵，n行，dim列，dim为数据维数，每一行为一个训练样本，每一列为一个特征
    :param y: 训练数据的标签列表，一维numpy.array(string)
    :param cost: 大于0的浮点数，SVM超参数，alpha_i的上界
    :param degree: 大于1的整数，多项式核函数的次数
    :return list: SVM多分类器
    """
    kernelf: Callable = get_kernel(kernel, kargs)
    n, dim = X.shape
    nclass = len(np.unique(y))
    classes = ovo(y)  # 拆分成多个一对一元组
    model = []
    print(f"此项任务总共需要 {len(classes)} 个分类器......")
    for i in range(len(classes)):
        A = X[y == classes[i][0], :]
        B = X[y == classes[i][1], :]
        yA = np.ones((A.shape[0], 1))
        yB = -np.ones((B.shape[0], 1))
        X_tmp = np.vstack((A, B))
        y_tmp = np.vstack((yA, yB))
        print(
            f"训练分类器: {i+1}/{len(classes)} ['{classes[i][0]}', '{classes[i][1]}']")
        model.append(nlsvm_solve(
            X_tmp, y_tmp, (classes[i][0], classes[i][1]), cost, kernelf))
    print("训练完成!")
    print(f"训练样本数: {n}；维数: {dim}；类别数: {nclass}")
    return model


def predict_1dim(model: list[Callable], x: np.ndarray) -> str:
    """
    单个样本的预测
    :param model: SVM多分类器，由函数闭包组成的列表
    :parma x: 1维np.array
    :return str: x 的预测类
    """
    classes = list(map(lambda submodel: submodel(x), model))
    unique_class, counts = np.unique(
        classes, return_counts=True)  # 计算每个类别出现的次数
    return unique_class[np.argmax(counts)]  # 返回出现次数最多的类别


def predict(model: list[Callable], X: np.ndarray) -> np.ndarray:
    """
    多个样本的预测
    :param model: SVM多分类器，由函数闭包组成的列表
    :parma X: 要预测数据的特征矩阵，np.array
    :return np.ndarray: 各个样本的预测分类
    """
    n = X.shape[0]  # 样本数
    print(f"测试样本数: {n}")
    predict_func = partial(predict_1dim, model)
    result = list(map(lambda x: predict_func(x), X))
    return np.array(result)


def accuracy(y_pred: np.ndarray, y_test: np.ndarray) -> float:
    """
    计算测试样本的准确率
    :param y_pred: 一维预测标签，类型为np.ndarray[str]
    :param y_test: 一维测试标签，类型为np.ndarray[str]
    :return float: 准确率
    """
    return sum(y_pred == y_test) / len(y_pred)


if __name__ == '__main__':
    from datetime import datetime
    import os
    os.chdir(os.path.dirname(__file__))
    from sklearn.model_selection import train_test_split

    from sklearn import datasets
    starttime = datetime.now()
    lris_df = datasets.load_iris()
    X = lris_df.data
    y = lris_df.target.astype(int).astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    # 高斯核函数，sigma=1.0
    print("对鸢尾花数据集进行训练......")
    model = nlsvm(X_train, y_train, cost=10.0, kernel="rbf", sigma=1.0)
    print("正在进行分类器测试......")
    y_predict = predict(model, X_test)
    print(f"测试集准确率: {accuracy(y_predict, y_test)*100:.2f}%")
    endtime = datetime.now()
    print("总共用时: ", (endtime - starttime).seconds, "秒")

    # import pandas as pd
    # data_train = pd.read_csv(
    #     "zip.train", delimiter=" ", header=None).to_numpy()
    # data_test = pd.read_csv("zip.test", delimiter=" ",
    #                         header=None).to_numpy()
    # X_train = data_train[:, 1:].astype(float)
    # y_train = data_train[:, 0].astype(int).astype(str)
    # X_test = data_test[:, 1:].astype(float)
    # y_test = data_test[:, 0].astype(int).astype(str)

    # starttime = datetime.now()
    # print("对手写数字数据集进行训练......")
    # model = nlsvm(X_train, y_train, cost=1.0, kernel="poly", degree=3)
    # print("正在进行分类器测试......")
    # y_predict = predict(model, X_test)
    # print(f"测试集准确率: {accuracy(y_predict, y_test)*100:.2f}%\n")
    # endtime = datetime.now()
    # print("总共用时: ", (endtime - starttime).seconds, "秒\n")

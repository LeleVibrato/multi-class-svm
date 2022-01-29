# -*- coding: utf-8 -*-
# @时间 : 2022年1月1日
# @作者 : sam255
# @文件名 : nlsvm.py
# @软件 : SVM多分类器

import numpy as np
from scipy import sparse
import osqp
from functools import partial


def poly_kernel(X1, X2, degree):
    """多项式核函数
    :param X1: 第一个向量
    :param X2: 第二个向量
    :param degree: 多项式的次数
    :return : 两个向量的内积
    :rtype : 浮点数
    """
    return (1+np.dot(X1, X2))**degree


def gaussian_kernel(X1, X2, sigma):
    """高斯核函数
    :param X1: 第一个向量
    :param X2: 第二个向量
    :param sigma: 高斯核函数的sigma
    :return : 两个向量的内积
    :rtype : 浮点数
    """
    return np.exp(-np.linalg.norm(X1-X2)**2 / (2 * (sigma ** 2)))


def nlsvm_solve(X, y, classes, C, degree):
    """训练一个以多项式为核函数的SVM二分类器
    :param X: 训练数据的特征矩阵，n行，dim列，dim为数据维数，每一行为一个训练样本，每一列为一个特征
    :param y: 训练数据的标签矩阵，n行，1列，取值为-1或1
    :param classes: 长度为2的字符串元组，表示当前用于训练的二分类
    :param C: SVM超参数，alpha_i的上界
    :param degree: 多项式核函数的次数
    :return : SVM二分类器
    :rtype: 函数闭包
    """
    n = X.shape[0]  # 样本个数

    # 计算内积
    inner_product = np.ones((n, n))
    for i in range(n):
        for j in range(i+1):
            inner_product[i, j] = inner_product[j, i] = poly_kernel(X[j, :],
                                                                    X[i, :], degree)

    # 定义二次规划参数矩阵
    q = -np.ones(n)
    P = 0.5 * y @ y.T * inner_product
    P[np.diag_indices_from(P)] = 2 * P[np.diag_indices_from(P)]
    A = np.vstack((np.diag(np.ones(n)), y.T))
    lb = np.zeros(n+1)
    ub = C * np.ones(n+1)
    ub[-1] = 0.0

    # 求解二次规划
    prob = osqp.OSQP()
    prob.setup(sparse.csc_matrix(P), q,
               sparse.csc_matrix(A), lb, ub, verbose=False)
    res = prob.solve()
    alpha = res.x

    # 寻找支持向量的索引值
    svs_ind = np.where(np.logical_and(0.0 < alpha, alpha < C))[0]

    # 计算 beta0（取平均值）
    beta0sum = 0.0
    for j in svs_ind:
        beta0sum += 1.0 / y[j, 0] - sum(alpha[i] * y[i, 0] * inner_product[j, i]
                                        for i in range(n))
    beta0 = beta0sum / len(svs_ind)  # 取平均值

    def nlsvm_classifer(x):
        """SVM 二分类器，作为函数闭包被上一级函数返回
        :param x: 1维np.array，长度为dim
        :return : 样本 x 的预测类别
        :rtype : 字符串
        """
        result = beta0 + \
            sum(alpha[i] * y[i, 0] * poly_kernel(x, X[i, :], degree)
                for i in range(n))
        if result >= 0.0:
            return classes[0]
        else:
            return classes[1]

    return nlsvm_classifer


def ovo(y):
    """将类别拆分成多个一对一元组
    :param y: 一维的字符串列表，代表各个训练样本的分类
    :return : 各个分类器要进行的二分类类别
    :rtype : 二元元组列表，元组元素类型为字符串
    """
    unique_cate = np.unique(y)
    unique_cate.sort()
    cate_num = len(unique_cate)  # 类别数
    print("类别: ", unique_cate)
    return [(unique_cate[i], unique_cate[j])
            for i in range(cate_num) for j in range(i+1, cate_num)]


def nlsvm(X, y, C=10.0, degree=3):
    """模型训练的接口函数
    :param X: 训练数据的特征矩阵，n行，dim列，dim为数据维数，每一行为一个训练样本，每一列为一个特征
    :param y: 训练数据的标签列表，一维numpy.array(string)
    :param C: 大于0的浮点数，SVM超参数，alpha_i的上界
    :param degree: 大于1的整数，多项式核函数的次数
    :return : SVM多分类器
    :rtype: 由函数闭包组成的列表，每个函数闭包代表一个分类器
    """
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
            X_tmp, y_tmp, (classes[i][0], classes[i][1]), C, degree))
    print("训练完成!")
    print(f"训练样本数: {n}；维数: {dim}；类别数: {nclass}")
    return model


def predict_1dim(model, x):
    """ 单个样本的预测
    :param model: SVM多分类器，由函数闭包组成的列表
    :parma x: 1维np.array
    :return : x 的预测类
    :rtype: 字符串
    """
    classes = list(map(lambda submodel: submodel(x), model))
    unique_class, counts = np.unique(
        classes, return_counts=True)  # 计算每个类别出现的次数
    return unique_class[np.argmax(counts)]  # 返回出现次数最多的类别


def predict(model, X):
    """ 多个样本的预测
    :param model: SVM多分类器，由函数闭包组成的列表
    :parma X: 要预测数据的特征矩阵，np.array
    :return : 各个样本的预测分类
    :rtype: 由字符串组成的列表
    """
    n = X.shape[0]  # 样本数
    print(f"测试样本数: {n}")
    predict_func = partial(predict_1dim, model)
    result = list(map(lambda x: predict_func(x), X))
    return np.array(result)


def accuracy(y_pred, y_test):
    """计算测试样本的准确率
    :param y_pred: 一维预测标签，类型为np.array(string)
    :param y_test: 一维测试标签，类型为np.array(string)
    :return : 准确率
    :rtype : 0到1之间的浮点数
    """
    return sum(y_pred == y_test) / len(y_pred)


if __name__ == '__main__':
    from datetime import datetime
    import os
    os.chdir(os.path.dirname(__file__))
    from sklearn.model_selection import train_test_split

    from sklearn import datasets
    starttime = datetime.now()
    print("对鸢尾花数据集进行训练......")
    lris_df = datasets.load_iris()
    X = lris_df.data
    y = lris_df.target.astype(int).astype(str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    model = nlsvm(X_train, y_train, C=10.0, degree=5)
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
    # model = nlsvm(X_train, y_train, C=1.0, degree=3)
    # print("正在进行分类器测试......")
    # y_predict = predict(model, X_test)
    # print(f"测试集准确率: {accuracy(y_predict, y_test)*100:.2f}%\n")
    # endtime = datetime.now()
    # print("总共用时: ", (endtime - starttime).seconds, "秒\n")

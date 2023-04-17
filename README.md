# multi-class-svm
通过求解SVM的对偶二次规划问题实现了一个支持核函数技巧的多分类支持向量机

其他说明
1. 实现了选择核函数类型的接口
2. demo使用的是鸢尾花数据集，可以直接执行命令```python nlsvm.py```
3. ```zip.train```和```zip.test```分别为手写数字数据集的训练样本和测试样本
4. 使用的二次规划求解器为```osqp```
5. 测试环境为```python 3.9.7```, 要求不低于```python 3.9```
6. 开源协议使用```MIT License```

example：
```python
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
```

output:
```
对鸢尾花数据集进行训练......
类别:  ['0' '1' '2']
此项任务总共需要 3 个分类器......
训练分类器: 1/3 ['0', '1']
训练分类器: 2/3 ['0', '2']
训练分类器: 3/3 ['1', '2']
训练完成!
训练样本数: 90；维数: 4；类别数: 3
正在进行分类器测试......
测试样本数: 60
测试集准确率: 96.67%
总共用时:  0 秒
```

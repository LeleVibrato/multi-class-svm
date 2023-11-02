# multi-class-svm
Implemented a multi-class support vector machine with kernel function tricks by solving the dual quadratic programming problem of SVM.

Additional Information:
1. Implemented an interface for selecting the type of kernel function.
2. The demo uses the Iris flower dataset and can be executed directly with the command ```python nlsvm.py```.
3. ```zip.train``` and ```zip.test``` are the training samples and test samples of the handwritten digit dataset, respectively.
4. The quadratic programming solver used is ```osqp```
5. The testing environment is ```python 3.9.7```, and it requires a minimum of```python 3.9```
6. The open-source license used is ```MIT License```

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

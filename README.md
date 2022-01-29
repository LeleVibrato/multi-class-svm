# multi-class-svm
通过求解SVM二次规划问题实现了一个支持核函数技巧的多分类SVM

其他说明
1. 默认使用多项式核函数，目前没有实现选择核函数类型的接口
2. 演示demo使用的是鸢尾花数据集，可以直接执行命令```python nlsvm.py```
3. ```zip.train```和```zip.test```分别为手写数字数据集的训练样本和测试样本
4. 使用的二次规划求解器为```osqp```
5. 本项目为本人<统计学习>课程大作业，只为学习目的创建

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
    iris_df = datasets.load_iris()  # Load the Iris dataset
    X = iris_df.data
    y = iris_df.target.astype(int).astype(str)
    
    # Split the dataset into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    
    print("Training on the Iris dataset...")
    # Use the Gaussian kernel function with sigma=1.0
    model = nlsvm(X_train, y_train, cost=10.0, kernel="rbf", sigma=1.0)
    
    print("Testing the classifier...")
    y_pred = predict(model, X_test)
    print(f"Test set accuracy: {accuracy(y_pred, y_test)*100:.2f}%")
    endtime = datetime.now()
    print("Total time: ", (endtime - starttime).seconds, "seconds")
```

output:
```
Training on the Iris dataset...
classes:  ['0' '1' '2']
A total of 3 classifiers are required for this task...
Training classifier: 1/3 ['0', '1']
Training classifier: 2/3 ['0', '2']
Training classifier: 3/3 ['1', '2']
Training complete!
Number of training samples: 90; Dimension: 4; Number of classes: 3
Testing the classifier...
Number of test samples: 60
Test set accuracy: 98.33%
Total time:  0 seconds
```

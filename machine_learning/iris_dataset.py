from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data[0])
print(iris.target[0])
removed = [0, 50, 100]
new_target = np.delete(iris.target, removed)
new_data = np.delete(iris.data, removed, axis=0)
print('original labels as per data', iris.target[removed])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_data, new_target)
prediction = clf.predict(iris.data[removed])
print('labels as per the algorithm', prediction)

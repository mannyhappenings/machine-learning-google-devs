from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# For visualization
from sklearn.externals.six import StringIO
import pydot

iris_dataset = load_iris()

test_idx = [0, 50, 100]

# Training data
train_target = np.delete(iris_dataset['target'], test_idx)
train_data = np.delete(iris_dataset['data'], test_idx, axis=0)

DTC = tree.DecisionTreeClassifier()

classifier_tree = DTC.fit(train_data, train_target)

print("Predictions", classifier_tree.predict(iris_dataset['data'][test_idx]))
print("Actual", iris_dataset['target'][test_idx])


# Visualization
dot_data = StringIO()

tree.export_graphviz(classifier_tree,
					 out_file=dot_data,
					 feature_names=iris_dataset['feature_names'],
					 class_names=iris_dataset['target_names'],
					 filled=True,
					 rounded=True,
					 impurity=False)

graph = pydot.graph_from_dot_data(dot_data.getvalue())
# graph[0].write_pdf('decision_tree.pdf')
print(dot_data.getvalue())
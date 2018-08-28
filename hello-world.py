from sklearn import tree

features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]

dt_classifier = tree.DecisionTreeClassifier()

trained_classifier = dt_classifier.fit(features, labels)

print(trained_classifier.predict([[150, 0]]))
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# [height, weight, shoe_size] - general data
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]
Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

Z = [[190, 70, 43]]

# build different classifiers

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict(Z)
print('Decision tree: {}'.format(prediction) )

neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
neigh.fit(X, Y) 

prediction = neigh.predict(Z)
print('KNeighbors: {}'.format(prediction) )


clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
clf.fit(X, Y)
prediction = clf.predict(Z)
print('Random Forest: {}'.format(prediction) )

clf = SVC(gamma='auto')
clf.fit(X, Y)
prediction = clf.predict(Z)
print('Support Vector Classification: {}'.format(prediction) )

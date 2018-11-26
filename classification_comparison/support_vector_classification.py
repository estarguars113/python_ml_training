import numpy as np
from sklearn.svm import SVC

# [height, weight, shoe_size] - train data
X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])

Y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male'])

clf = SVC(gamma='auto')
clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])

print(prediction)
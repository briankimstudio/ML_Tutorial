
#
# Linear Regression Classifier
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#
# To generate consistant experimental results
#
np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

my_dataset = pd.DataFrame(X)
my_dataset['Target'] = y

#
# Check data structure, # of rows, # of columns
#
print('\nData structure\n')
my_dataset.info()

#
# Check value of data
#
print('\nData value\n')
print(my_dataset.head())

clf = LinearRegression()
###############################################################################
#
# 2. Training
#
###############################################################################

print('\nTraining...\n')
clf.fit(X, y)
print(clf.score(X, y))

print('\nCoefficient')
print(clf.coef_)

###############################################################################
#
# 3. Estimating
#
###############################################################################

print('\nPredicting...\n')
pred = clf.predict(np.array([[3, 5]]))

print(pred)
###############################################################################
#
# 4. Evaluating
#
###############################################################################

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# disp.plot()
# plt.show()

# print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
# print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
# print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
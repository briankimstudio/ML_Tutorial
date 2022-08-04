
#
# Naive Bayes Classifier
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#
# To generate consistant experimental results
#
np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################
print('\nData preparation...\n')

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')
clf = GaussianNB()
clf.fit(X_train, y_train)

###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\nPredicting...\n')

y_pred = clf.predict(X_test)

###############################################################################
#
# 4. Evaluating
#
###############################################################################
print('\Evaluating...\n')

print("Number of mislabeled points out of a total %d points : %d"
      % (X_test.shape[0], (y_test != y_pred).sum()))

cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
disp.plot()
plt.show()

print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
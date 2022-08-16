#
# Support Vector Machine Classifier
#

from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#
# To generate consistant experimental results
#
np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################

#
# Read dataset from library
#
cancer = datasets.load_breast_cancer()

print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)

# print data(feature)shape
cancer.data.shape

# print the cancer data features (top 5 records)
print(cancer.data[0:5])

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target[0:5])

#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=109)
print(f'Train :{X_train.shape}, Test :{X_test.shape}')

#
# 
#

kernel_type = 'linear'
clf = svm.SVC(kernel = kernel_type)

###############################################################################
#
# 2. Training
#
###############################################################################

clf.fit(X_train, y_train)

###############################################################################
#
# 3. Estimating
#
###############################################################################

#
# Predict with test set
#
y_pred = clf.predict(X_test)

###############################################################################
#
# 4. Evaluating
#
###############################################################################

cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

metrics.plot_roc_curve(clf, X_test, y_test) 
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
plt.show()

print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')

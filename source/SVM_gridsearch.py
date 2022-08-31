#
# Support Vector Machine Classifier - Binary Classification
#

from sklearn import datasets
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
raw_dataset = datasets.load_breast_cancer()
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target

#
# Convert numeric value to string in class
#
my_dataset['class'].replace([0,1],raw_dataset.target_names, inplace=True)

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

#
# Count data per class
#
print('\nCount by class\n')
print(my_dataset['class'].value_counts())

#
# Descriptive statistics
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

X = my_dataset.drop('class',axis=1)
y = my_dataset['class']

#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=5)
print(f'\nTraining set : {X_train.shape}, Test set : {y_train.shape}\n')

#
# Set hyperparameter
#
parameters = {'kernel': ('linear','rbf'), 'C':[1,10]}

svc = svm.SVC(probability=True, random_state=0)

#
# Hyperparameter optimization
#
clf_grid = GridSearchCV(svc, parameters)

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print(f'\nTraining...\n')

clf_grid.fit(X_train, y_train)
print(f'Training score : {clf_grid.score(X_train, y_train)}')
print(f'Best params : {clf_grid.best_params_}')
print(f'Cross validation results : {pd.DataFrame(clf_grid.cv_results_).T}')

#
# Select estimator trained with best parameters
#
clf = clf_grid.best_estimator_

###############################################################################
#
# 3. Estimating
#
###############################################################################

#
# Predict with test set
#
print('\nPredicting...\n')
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)

###############################################################################
#
# 4. Evaluating
#
###############################################################################

cm = confusion_matrix(y_test, y_pred)
print(cm)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
# disp.plot()
# plt.show()

# metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
# plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
# plt.show()

#
# Check whether prediction is correct
#
results = pd.DataFrame(y_test.array, columns=['truth'])
results['predict'] = y_pred
results['result']  = results.apply(lambda row: 'correct' if row.truth == row.predict else 'wrong', axis=1)
print(f'\n{results}\n')
print(results['result'].value_counts())

#
# Check performance of the model
#
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob[:,1])}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred, pos_label="malignant")}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred, pos_label="malignant")}')

#
# Linear Regression Classifier
#

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import seaborn as sns
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

raw_dataset = datasets.load_diabetes()

# Create pandas dataframe
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target

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
# Descriptive statistics
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

plt.subplots(figsize=(7,7))
sns.heatmap(my_dataset.corr(), cmap='RdYlBu', annot=True)
plt.show()

#
#
#
X = my_dataset.drop('class',axis=1)
y = my_dataset['class']


#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f'\nTraining set : {X_train.shape}, Test set : {y_train.shape}\n')

clf = LinearRegression()

###############################################################################
#
# 2. Training
#
###############################################################################

print('\nTraining...\n')
clf.fit(X, y)
print(f'Training score : {clf.score(X_train, y_train)}')
print(f'Coefficient : {clf.coef_}')

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

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# disp.plot()
# plt.show()

# plt.scatter(X_test, y_test, color="black")
# plt.plot(X_test, y_pred, linewidth=2)

# plt.xticks(())
# plt.yticks(())

# plt.show()

print(f'MSE : {mean_squared_error(y_test, y_pred)}')
print(f'R2  : {r2_score(y_test, y_pred)}')

# print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
# print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
# print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
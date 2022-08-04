
#
# Linear Regression Classifier
#

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns

#
# To generate consistant experimental results
#
np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################

my_dataset = pd.read_csv('https://raw.githubusercontent.com/datagy/data/main/insurance.csv')

print(my_dataset.head())

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
# Check correlation
#
print(my_dataset.corr())

sns.pairplot(my_dataset)
# plt.show()

# Plotting a pairplot of your DataFrame
sns.pairplot(my_dataset, hue='smoker')
# plt.show()

sns.relplot(data=my_dataset, x='age', y='charges', hue='smoker')
# plt.show()

non_smokers = my_dataset[my_dataset['smoker'] == 'no']
print(non_smokers.corr())

my_dataset['smoker_int'] = my_dataset['smoker'].map({'yes':1,'no':0})

X = my_dataset[['age','bmi','smoker_int']]
y = my_dataset['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, train_size=0.3)

clf = LinearRegression()
###############################################################################
#
# 2. Training
#
###############################################################################

print('\nTraining...\n')
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))

print('\nCoefficient')
print(clf.coef_)


###############################################################################
#
# 3. Estimating
#
###############################################################################

print('\nPredicting...\n')
y_pred = clf.predict(X_test)


# print(y_pred)
###############################################################################
#
# 4. Evaluating
#
###############################################################################
print('\nEvaluating...\n')
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'R2           : {r2}')
print(f'RMSE         : {rmse}')

print(f'Intercept    : {clf.intercept_}')
print(f'Coefficients : {clf.coef_}')

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# disp.plot()
# plt.show()

# print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
# print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
# print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
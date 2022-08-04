#
# Data Preparation
#
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#
# Read dataset from library
#
raw_data = datasets.load_wine()
my_dataset = pd.DataFrame(raw_data.data)
my_dataset['Class'] = raw_data.target
#
# Conver numeric value to string in class
#

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
print(my_dataset['Class'].value_counts())

#
#
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

#
# Draw scatter plot
#
# sns.pairplot(my_dataset,hue='Class',markers='+')

#
#
#
X = my_dataset.drop('Class',axis=1)
y = my_dataset['Class']

#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)
print(f'\nTrain set : {X_train.shape}, Test set : {y_train.shape}\n')
#
# Set 5 neighours
#
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

#
# Train with training set
#
print(f'\nTraining K={k}...\n')
knn.fit(X_train, y_train)

#
# Predict with test set
#
print('\nPredicting...\n')
y_pred = knn.predict(X_test)

#
# Check whether prediction is correct
#
results = pd.DataFrame(y_test.array, columns=['Truth'])
results['Predict'] = y_pred
results['Result'] = results.apply(lambda row: 'Correct' if row.Truth == row.Predict else 'Wrong', axis=1)
print(f'\n{results}\n')
print(results['Result'].value_counts())

#
# Check accuracy
#
print(f'\nAccuracy : {metrics.accuracy_score(y_test,y_pred)}\n')

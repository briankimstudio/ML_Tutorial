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
raw_dataset = datasets.load_iris()
# cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'] 

# Create pandas dataframe
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target


#
# Convert numeric value to string in class
#
my_dataset['class'].replace([0,1,2],raw_dataset.target_names, inplace=True)

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
#
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

#
# Draw scatter plot
#
sns.pairplot(my_dataset,hue='class',markers='+')

#
#
#
X = my_dataset.drop('class',axis=1)
y = my_dataset['class']

#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=5)
print(f'\nTrain set : {X_train.shape}, Test set : {y_train.shape}\n')
#
# Set 5 neighours
#
k = 5
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
results = pd.DataFrame(y_test.array, columns=['truth'])
results['predict'] = y_pred
results['result'] = results.apply(lambda row: 'correct' if row.truth == row.predict else 'wrong', axis=1)
print(f'\n{results}\n')
print(results['result'].value_counts())

#
# Check accuracy
#
print(f'\nAccuracy : {metrics.accuracy_score(y_test,y_pred)}\n')
print(f'Precision : {metrics.precision_score(y_test, y_pred, average=None)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred, average=None)}')
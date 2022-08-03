#
# Decision Tree classifier
#

from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
#from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#
# To generate consistant experimental results
#
np.random.seed(0)

#
# Read dataset from library
#
iris = datasets.load_iris()
cols = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width'] 
my_dataset = pd.DataFrame(iris.data, columns=cols)
my_dataset['Class'] = iris.target

#
# Convert numeric value to string in class
#
my_dataset['Class'].replace([0,1,2],['Setosa','Versicolor','Virginica'], inplace=True)

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
sns.pairplot(my_dataset,hue='Class',markers='+')

#
# Save plot as PNG file
#
plt.savefig('Iris_scatter_plot.png')

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
# Decision Tree Classifier 
#
clf = DecisionTreeClassifier()

#
# Train with training set
#
print(f'\nTraining...\n')
clf.fit(X_train, y_train)

#
# Predict with test set
#
print('\nPredicting...\n')
y_pred = clf.predict(X_test)

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


#
# Random Forest Classifier - Multiclass Classification
#

from sklearn import datasets
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
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
print('\nData preparation...\n')

#
# Read dataset from library
#
raw_dataset = datasets.load_iris()
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
# Descriptive statistics
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

#
# Draw scatter plot
#
sns.pairplot(my_dataset,hue='class',markers='+')
# iris = datasets.load_iris()
# print(iris.target_names)

# # print the names of the four features
# print(iris.feature_names)

# data=pd.DataFrame({
#     'sepal length':iris.data[:,0],
#     'sepal width':iris.data[:,1],
#     'petal length':iris.data[:,2],
#     'petal width':iris.data[:,3],
#     'species':iris.target
# })
# data.head()

# X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
# y=data['species']  # Labels

X = my_dataset.drop('class',axis=1)
y = my_dataset['class']

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(f'\nTraining set : {X_train.shape}, Test set : {y_train.shape}\n')

#
# Set hyperparameter
#
n = 100
clf = RandomForestClassifier(n_estimators = n)

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print('\nTraining...\n')
clf.fit(X_train, y_train)
print(f'Training score : {clf.score(X_train, y_train)}')

###############################################################################
#
# 3. Estimating
#
###############################################################################
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
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
disp.plot()
plt.show()

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
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob, multi_class="ovo")}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision  : {metrics.precision_score(y_test, y_pred, average=None)}')
print(f'Recall     : {metrics.recall_score(y_test, y_pred, average=None)}')
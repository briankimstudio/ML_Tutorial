
#
# Random Forest Classifier
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

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
iris = datasets.load_iris()
print(iris.target_names)

# print the names of the four features
print(iris.feature_names)

data=pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
data.head()

X=data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
y=data['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

clf=RandomForestClassifier(n_estimators=100)

###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')


#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)


###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\nPredicting...\n')
y_pred=clf.predict(X_test)

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

print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
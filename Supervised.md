## Supervised Machine Learning

1. Linear Regression
2. Logistic Regression
3. K-Nearest Negihbors (KNN)
4. Support Vector Machine (SVM)

Supervised machine learning requires input data with features and label. Typical applications are classification(identifying category such as cat or dog) and regression(predicting continuous value such as house price). 

### 1. Linear Regression

For social science research, regression is used to find relationship between independent variable and dependent variable. As a result, coefficient and p value are inspected to accept or reject research hypothesis. Normally, social science research stops here. 

However, in the area of machine learning, the important part just starts from here. After fitting the data, we save coefficients and intersepts as a computer file, which is called a 'trained machine learning model' Then, new data are fed into the model to predict value of dependent variable.

### 2. Logistic Regression

Logistic regression is a binary classification.

### 3. K-Nearest Negihbors (KNN)

Hyperparameter

__k = 5__

Model
```
#
# Set hyperparameter
#
k = 5
knn = KNeighborsClassifier(n_neighbors = k)

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print(f'\nTraining K={k}...\n')
knn.fit(X_train, y_train)

###############################################################################
#
# 3. Estimating
#
###############################################################################

#
# Predict with test set
#
print('\nPredicting...\n')
y_pred = knn.predict(X_test)
y_prob = knn.predict_proba(X_test)

###############################################################################
#
# 4. Evaluating
#
###############################################################################

#
# Check whether prediction is correct
#
results = pd.DataFrame(y_test.array, columns=['truth'])
results['predict'] = y_pred
results['result']  = results.apply(lambda row: 'correct' if row.truth == row.predict else 'wrong', axis=1)
print(f'\n{results}\n')
print(results['result'].value_counts())

#
# Check accuracy
#
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob, multi_class="ovo")}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred, average=None)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred, average=None)}')
```

Results

Summary of prediction results of KNN model are presented below. It made 28 correct predictions and 2 wrong predictions. AUC score and accuracy is for the model, whereas precision and recall are for each class: setosa, versicolor, and virginica.

__k = 5__
```
correct    28
wrong       2
Name: result, dtype: int64

AUC score : 0.997245179063361
Accuracy  : 0.9333333333333333
Precision : [1.         1.         0.84615385]
Recall    : [1.         0.81818182 1.        ]
```

Let's modify the `k` to other value and check the results again to find out the impact of the `k` in the performance of the model. In the case of `k = 10`, the number of correct prediction is increased to 29 and wrong prediction is decreased to 1. Acccordingly, other performance indicators are improved as well.

__k = 10__

```
correct    29
wrong       1
Name: result, dtype: int64

AUC score : 0.9986225895316805
Accuracy  : 0.9666666666666667
Precision : [1.         1.         0.91666667]
Recall    : [1.         0.90909091 1.        ]
```



### 4. Support Vector Machine (SVM)

Model
```
```

Results
```
```

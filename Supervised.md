## Supervised Machine Learning

### Summary of methods

| Type | Method | Dataset |
| --- | --- | --- |
| Regression | Linear Regression | `diabetes` |
| Binary Classification | Logistic Regression | `breast cancer` |
| Binary Classification | Support Vector Machine | `breast cancer` |
| Multiclass Classification | K-Nearest Negihbors | `iris` |
| Multiclass Classification | Decision Tree | `iris` |
| Multiclass Classification | Random Forest | `iris` |

Supervised machine learning requires input data with features and label. Typical applications are classification(identifying category such as cat or dog) and regression(predicting continuous value such as house price). 

### Regression

### 1. Linear Regression

For social science research, regression is used to find relationship between independent variable and dependent variable. As a result, coefficient and p value are inspected to accept or reject research hypothesis. Normally, social science research stops here. 

However, in the area of machine learning, the important part just starts from here. After fitting the data, we save coefficients and intersepts as a computer file, which is called a 'trained machine learning model' Then, new data are fed into the model to predict value of dependent variable.

### Binary Classification

### 2. Logistic Regression

Logistic regression is a binary classification.

Model
```
#
# Set hyperparameter
#
clf = LogisticRegression(random_state=0)

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print(f'\nTraining...\n')
clf.fit(X_train, y_train)
print(f'Training score : {clf.score(X_train, y_train)}')

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
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
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob[:,1])}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred, pos_label="malignant")}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred, pos_label="malignant")}')
```

Results
```
Predicting...

[[107   3]
 [  3  58]]

correct    165
wrong        6
Name: result, dtype: int64

AUC score : 0.9923994038748137
Accuracy  : 0.9649122807017544
Precision : 0.9508196721311475
Recall    : 0.9508196721311475
```
### Multiclass Classification

### 3. Support Vector Machine (SVM)

Model
```
#
# Set hyperparameter
#
kernel_type = 'linear'
clf = svm.SVC(kernel = kernel_type, probability=True, random_state=0)

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print(f'\nTraining...\n')
clf.fit(X_train, y_train)
print(f'Training score : {clf.score(X_train, y_train)}')

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

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
disp.plot()
plt.show()

metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test)
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
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
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob[:,1])}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred, pos_label="malignant")}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred, pos_label="malignant")}')
```

Results
```
Predicting...

[[108   2]
 [  1  60]]

correct    168
wrong        3
Name: result, dtype: int64

AUC score : 0.9895678092399404
Accuracy  : 0.9824561403508771
Precision : 0.967741935483871
Recall    : 0.9836065573770492
```

### 4. K-Nearest Negihbors (KNN)

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
Predicting...

[[ 8  0  0]
 [ 0  9  2]
 [ 0  0 11]]

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
Predicting...

[[ 8  0  0]
 [ 0 10  1]
 [ 0  0 11]]
 
correct    29
wrong       1
Name: result, dtype: int64

AUC score : 0.9986225895316805
Accuracy  : 0.9666666666666667
Precision : [1.         1.         0.91666667]
Recall    : [1.         0.90909091 1.        ]
```

The question here is what is the best value for the `k`. How we can find it?

### Decision Tree

Model
```
#
# Decision Tree Classifier 
#
clf = DecisionTreeClassifier()

###############################################################################
#
# 2. Training
#
###############################################################################

#
# Train with training set
#
print(f'\nTraining...\n')
clf.fit(X_train, y_train)
print(f'Training score : {clf.score(X_train, y_train)}')

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

#
# Check whether prediction is correct
#
cm = confusion_matrix(y_test, y_pred)
print(cm)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
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
# Check accuracy
#
print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_prob, multi_class="ovo")}')
print(f'Accuracy  : {metrics.accuracy_score(y_test, y_pred)}')
print(f'Precision  : {metrics.precision_score(y_test, y_pred, average=None)}')
print(f'Recall     : {metrics.recall_score(y_test, y_pred, average=None)}')
```

Results
```
Predicting...

[[ 8  0  0]
 [ 0  9  2]
 [ 0  0 11]]

correct    28
wrong       2
Name: result, dtype: int64

AUC score : 0.9545454545454546
Accuracy  : 0.9333333333333333
Precision  : [1.         1.         0.84615385]
Recall     : [1.         0.81818182 1.        ]
```

### 5. Random Forest

```
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
```

Results
```
Predicting...

[[ 8  0  0]
 [ 0 10  1]
 [ 0  2  9]]

correct    27
wrong       3
Name: result, dtype: int64

AUC score : 0.9944903581267218
Accuracy  : 0.9
Precision  : [1.         0.83333333 0.9       ]
Recall     : [1.         0.90909091 0.81818182] 
```
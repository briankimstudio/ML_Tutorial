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

However, in the area of machine learning, the important part just starts from here. After fitting the data, we save coefficients and intercept as a computer file, which is called a 'trained machine learning model.' Then, new data are fed into the model to predict value of dependent variable.

Model
```
#
# Set hyperparameter
#
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

print(f'MSE : {mean_squared_error(y_test, y_pred)}')
print(f'R2  : {r2_score(y_test, y_pred)}')
```

Results
```
Predicting...

MSE : 2864.449455303467
R2  : 0.5457325573378824
```
### Binary Classification

### 2. Logistic Regression

Among various types of regression, logistic regression is used when the dependent variable is a categorical type of either 0 or 1. So, it fits for solving binary classification problem.

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

### Performance comparison of binary classification models 

| | AUC Score | Accuracy |
|---|---:|---:|
| Logistic regression | 0.99 | 0.96 |  
| SVM | 0.98 | 0.98 |

### Multiclass Classification

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

### Performance comparison of multiclass classification models 

| | AUC Score | Accuracy |
|---|---:|---:|
| K-Nearest Neighbors  | 1.00 | 0.96 |  
| Decision Tree | 0.95 | 0.93 |
| Random Forest | 0.99 | 0.90 |

### Overfitting problem

Overfitting means the circumstance that the model performs well with the traning data but performs worse with the actual data. It happens when the model considered too much details of the data during the training phase. Therefor, the model works well for only training data and generalization of the model would be limited.

### K fold cross validation

Regarding overfitting, one of the cauase is inappropriate split of training and validation dataset. For example, if all outliers are in training dataset, then, the model can not perform well with validation dataset. Therefore, it is necessary to train the model with different combination of training and validation dataset repeatedly in order to minimize the impact of outliers or abnormal data.

```
print(f'\nTraining...\n')
for n in range(2,10,1):
    kfold = KFold(n_splits=n)
    scores = cross_val_score(clf, X_train, y_train, cv=kfold, n_jobs=-1)
    print(f'{n} fold cross validataion score(mean): {scores.mean()}')
    print(f'{n} fold cross validataion score(sd)  : {scores.std()}')
```
Results
```
Training...

2 fold cross validataion score(mean): 0.9296482412060302
2 fold cross validataion score(sd)  : 0.005025125628140725
3 fold cross validataion score(mean): 0.944691273638642
3 fold cross validataion score(sd)  : 0.017844033477176696
4 fold cross validataion score(mean): 0.9395707070707071
4 fold cross validataion score(sd)  : 0.025921086773443264
5 fold cross validataion score(mean): 0.9421202531645569
5 fold cross validataion score(sd)  : 0.03066493252634119
6 fold cross validataion score(mean): 0.9370948288858737
6 fold cross validataion score(sd)  : 0.036586912281119786
7 fold cross validataion score(mean): 0.9369405656999642
7 fold cross validataion score(sd)  : 0.04270733025003529
8 fold cross validataion score(mean): 0.9394897959183672
8 fold cross validataion score(sd)  : 0.052986505476384414
9 fold cross validataion score(mean): 0.9395061728395062
9 fold cross validataion score(sd)  : 0.05676710649997434
```

### Hyperparameter optimization

Typically, machine learning model needs specific set of conditions to perform training. These condition is called 'hyperparameter' and need to be specified by user. But, it is impossible to know what is the best value for hyperparameter, so, we need to repeat the training with different set of parameter, then, use the best one for the final training. 

### Gridsearch

Gridsearch is a feature to automate this process and pick the best hyperparameter for you. In this example, we used two hyperparameters: kernel and C. There are two types of kernel and two different value of C. Thus, overall we need to repeat the training four times in order to identify best combination of two hyperparameters({linear,1}, {linear,10}, {rbf,1}, {rbf,10}).

```
parameters = {'kernel': ('linear','rbf'), 'C':[1,10]}
svc = svm.SVC(probability=True, random_state=0)
clf_grid = GridSearchCV(svc, parameters)

#
# Train with training set
#
print(f'\nTraining...\n')
clf_grid.fit(X_train, y_train)
print(f'Training score : {clf_grid.score(X_train, y_train)}')
print(f'Best params : {clf_grid.best_params_}')
print(f'Grid search results : \n{pd.DataFrame(clf_grid.cv_results_).T}')
clf = clf_grid.best_estimator_
```

Results

After gridsearch, it indicates that hyperparameters of {linear,1} shows the best performance than the others. By default, it performs 5 fold cross validation and the results are presented from  `split0_test_score` to `split4_test_score`.

```
Training...

Training score : 0.9597989949748744
Best params : {'C': 1, 'kernel': 'linear'}
Grid search results :
                                              0                          1                              2                           3
mean_fit_time                          3.837402                   0.014985                      14.519973                    0.012566
std_fit_time                           1.437212                   0.004439                       3.664709                    0.001018
mean_score_time                             0.0                   0.002014                       0.000798                    0.002388
std_score_time                              0.0                   0.002466                       0.000978                    0.000493
param_C                                       1                          1                             10                          10
param_kernel                             linear                        rbf                         linear                         rbf
params             {'C': 1, 'kernel': 'linear'}  {'C': 1, 'kernel': 'rbf'}  {'C': 10, 'kernel': 'linear'}  {'C': 10, 'kernel': 'rbf'}
split0_test_score                        0.9625                      0.925                         0.9625                       0.925
split1_test_score                          0.95                     0.9625                         0.9375                       0.925
split2_test_score                         0.925                      0.875                         0.9125                         0.9
split3_test_score                      0.962025                   0.924051                       0.949367                    0.924051
split4_test_score                      0.886076                   0.810127                       0.911392                    0.835443
mean_test_score                         0.93712                   0.899335                       0.934652                    0.901899
std_test_score                         0.028923                   0.052554                       0.020159                    0.034577
rank_test_score                               1                          4                              2                           3
```

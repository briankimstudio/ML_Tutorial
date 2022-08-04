# Machine Learning Tutorial for Social Science Researcher

Originally, computer program executes rules provided by humans. But, in some application where there are too many data or too many factors, it is not easy for human to figure out the rules and give an order to computer. In this case, we provide data to computer, then let it figure out the rules by itself. This is the core concept of machine learning.

Social science resesarchers are familiar with various types of statistical analysis methods. Fortunately, these methods are very similar to popular machine learning methods.

__Statistical Analysis vs Machine Learning__

By comparing statistical analysis and machine learning, we can have clear view for these two fields.  

|         | Statistical Analysis | Machine Learning |
|--------:| :----------------: | :--------------: |
| User    | Sociel Scientist | Computer Scientist |
| Purpose | Hypothesis testing | Predicting       |
| Results | Coefficient, p vlaue | Trained model |
| Tools   | R, Python | Python, R |

__Types of Machine Learning Methods__

Unlike statistical analysis, the purpose of machine learning is to predict value which is unknown at the time of prediction. The value is either continous or binary. For example, for house price estimation, it is continouse number. But, for estimating whether the patient have a caner or not is binary value. The other types of problem is that there are lots of data and we woule like to cluster them into multiple groups with similar characteristics. 

| Binary Prediction | Multiclass Prediction | Clustering |
| :---------------: | :-------------------: | :--------: |
| Logistic regression | Linear Regression | K-means |
| Support Vector Machine | KNN | Topic Modeling |

This tutorial consists of three parts: 1) Data preparation, 2) Supervised Machine Learning, and 3) Unsupervised Machine Learning. `scikit-learn` is used for example source codes in this tutorial for simplicity and clarity. 

## Data Preparation

Data preparation covers folowing steps.

- Reading data
- Inspecting dimension
- Inspecting data type of columns
- Visualizing correlation among columns
 
## Supervised Machine Learning

Supervised machine learning requires input data with features and label. Typical applications of supervised machine learning are classification and prediction. 

### Linear Regression

For social science research, regression is used to find relationship between independent variable and dependent variable. As a result, coefficient and p value are inspected to accept or reject research hypothesis. Normally, social science research stops here. 

However, in the area of machine learning, the important part just starts from here. After fitting the data, we save coefficients and intersepts as a computer file, which is called a 'trained machine learning model' Then, new data are fed into the model to predict value of dependent variable.

### Logistic Regression

Logistic regression is a binary classification.

### K-Nearest Negihbors (KNN)

### Support Vector Machine (SVM)

## Unsupervised Machine Learning

Unsupervised machine learning requires input data with features but it has no label. Therefore, typical application of unsupervised machine learning is clustering.

### Topic Modeling

Topic modeling is a method to identify prevalent topics from dataset of natural language such as new papers, online reviews, or scholary articles.

### K means clustering

## Evaluation

### Confusion matrix

|   | Predicted  |Predicted|
|---|:---:|:---:|
| Actual | TP | FP |
| Actual       | TN | FN |


- Accuracy = TP+TN / TP+TN+FP+FN
- Precision = TP / TP+FP
- Recall = TP / TP+FN


### Receiver Operating Charasteristic(ROC) curve

In this curve, x axis indicates False Positive Rate(FPR) and y axis indicates True Positive Rate(TPR)

### Area Under the ROC Curve(AUC)

AUC is a frequently used indicator showing the performance of the model. It ranges from 0 to 1 and higher is better. 
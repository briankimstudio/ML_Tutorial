# Machine Learning Tutorial for Social Science Researcher

Originally, computer program executes rules provided by humans. But, in some application where there are too many data or too many factors, it is not easy for human to figure out the rules and give an order to computer. In this case, we provide data to computer, then let it figure out the rules by itself. This is the core concept of machine learning.

Social science resesarchers are familiar with various types of statistical analysis methods. Fortunately, these methods are very similar to popular machine learning methods.

__Statistical Analysis vs Machine Learning___

|         | Social Science     | Machine Learning |
|---------| :----------------: | :--------------: |
| Purpose | Hypothesis testing | Predicting       |

__Types of Machine Learning Methods__

Unlike statistical analysis, the purpose of machine learning is to predict value which is unknown at the time of prediction. The value is either continous or binary. For example, for house price estimation, it is continouse number. But, for estimating whether the patient have a caner or not is binary value. The other types of problem is that there are lots of data and we woule like to cluster them into multiple groups with similar characteristics. 

| Binary Prediction | Multiclass Prediction | Clustering |
| ----------------- | --------------------- | ---------- |
| Logistic regression | Linear Regression | K-means |
| Support Vector Machine | KNN | Topic Modeling |

This tutorial consists of three parts: 1) Data preparation, 2) Supervised Machine Learning, and 3) Unsupervised Machine Learning 

# Data Preparation

## Supervised Machine Learning

### Linear Regression

For social science research, regression is used to find relationship between independent variable and dependent variable. As a result, coefficient and p value are inspected to accept or reject research hypothesis. Normally, social science research stops here. 

However, in the area of machine learning, the important part just starts from here. After fitting the data, we save coefficients and intersepts as a computer file, which is called a 'trained machine learning model' Then, new data are fed into the model to predict value of dependent variable.

### Logistic Regression

### K-Nearest Negihbors (KNN)

### Support Vector Machine (SVM)

## Unsupervised Machine Learning

### Topic Modeling

Topic modeling is a method to identify prevalent topics from dataset of natural language such as new papers, online reviews, or scholary articles.

### K means clustering

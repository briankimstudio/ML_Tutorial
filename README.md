# Machine Learning Tutorial for Social Science Researcher

Originally, computer program executes rules provided by humans. But, in some application where there are too much data or too many factors, it is not easy for a human to figure out the rules and give an order to a computer. In this case, we provide data to a computer, then let it figure out the rules by itself. This is the core concept of machine learning.

Social science researchers are familiar with various types of statistical analysis methods. Fortunately, these methods are very similar to popular machine learning methods.

__Statistical Analysis vs Machine Learning__

By comparing statistical analysis and machine learning, we can have a clear view of these two fields.  

|         | Statistical Analysis | Machine Learning |
|--------:| :----------------: | :--------------: |
| User    | Social Scientist | Computer Scientist |
| Purpose | Hypothesis testing | Predicting       |
| Results | Coefficient, p vlaue | Trained model |
| Tools   | R, Python | Python, R |

__Purpose of Machine Learning Methods__

Unlike statistical analysis, the purpose of machine learning is to predict a value that is unknown at the time of prediction. The value is either continuous or binary. For example, in the case of house price estimation, it is a continuous number. But, for estimating whether the patient has cancer or not is abinary value. The other type of problem is that there are lots of data and we would like to cluster them into multiple groups with similar characteristics. 

| Binary Prediction | Multiclass Prediction | Clustering |
| :---------------: | :-------------------: | :--------: |
| Logistic regression | Linear Regression | K-means |
| Support Vector Machine | KNN | Topic Modeling |

This tutorial consists of three parts: 1) Data preparation, 2) Supervised Machine Learning, 3) Unsupervised Machine Learning, and 4) Evaluation of Machine Learning Model. `scikit-learn` is used for example source codes in this tutorial for simplicity and clarity. 

1. [Data Preparation](Data_preparation.md)
2. [Supervised Machine Learning](Supervised.md)
3. [Unsupervised Machine Learning](Unsupervised.md)
4. [Evaluation of Machine Learning Model](Evaluation.md)
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

This tutorial consists of three parts: 1) Data preparation, 2) Supervised Machine Learning, 3) Unsupervised Machine Learning, and 4) Evaluation of Machine Learning Model. `scikit-learn` is used for example source codes in this tutorial for simplicity and clarity. 

1. [Data Preparation](Data_preparation.md)
2. [Supervised Machine Learning](Supervised.md)
3. [Unsupervised Machine Learning](Unsupervised.md)
4. [Evaluation of Machine Learning Model](Evaluation.md)
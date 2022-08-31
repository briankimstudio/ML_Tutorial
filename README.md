# Machine Learning Tutorial for Social Science Researchers

Originally, computer program executes rules provided by humans. But, in some applications where there are too much data or too many factors, it is not easy for a human to figure out the rules and give an order to a computer. In this case, we provide data to a computer, then let it figure out the rules by itself. This is the core concept of machine learning.

Social science researchers are familiar with various types of statistical analysis methods. Fortunately, these methods are very similar to popular machine learning methods.

## Statistical Analysis vs Machine Learning

By comparing statistical analysis and machine learning, we can have a clear view of these two fields.  

|         | Statistical Analysis | Machine Learning |
|--------:| :----------------: | :--------------: |
| Major User    | Social scientist | Computer scientist |
| Purpose | Testing hypothesis | Predicting value       |
| Cause | Independent variables | Features       |
| Effect | Dependent variable | Label(Class)       |
| Output | Coefficients | Trained model, predictions |
| Evaluation | p-value, SRMR, GFI,NFI,CFI, RMSEA, ... | AUC, accuracy, precision, recall, ... |
| Platform   | R, jamovi, JASP, SPSS, ... | Python, R, ... |
| Package | ... | sciket learn, PyTorch, Tensorflow, Keras, ... |

## Purpose of Machine Learning

Unlike statistical analysis, the purpose of machine learning is to predict a value that is unknown at the time of prediction using the knowledge acquired via training. 

## Type of Machine Learning

Let's assume that we have a task that classifes cats and dogs, and we have a machine learning model called 'cat detector'. After inspecting an image, then it predicts whether the image is a cat or a dog just like in the figure below.

![overview](images/overview.png)

### Classification

The first type of problem is classification. The goal of this model is to classify binary values(cats or dogs) in the output. Thus, it is called a 'binary classification'. If there are more than two types of output, then it is called 'multiclass classification'. 

### Regression

The second type of problem is 'regression'. In the case of house price estimation, the goal of the model is to predict a continuous value, as opposed to predict a class.

### Clustering

The third type of problem is 'clustering'. Let's assume that there are lots of data and we would like to cluster them into multiple groups with similar characteristics. 

From machine learning's perspective, classification and regression belong to [supervised learning](Supervised.md), whereas clustering belongs to [unsupervised learning](Unsupervised.md).

|  | Binary Classification | Multiclass Classification | Regression |Clustering |
| ---: | :---------------: | :-------------------: | :---: | :--------: |
| Example | Predict cancer | Predict flowers species | Predict housr price | Separate photos by person |
| Output  | Yes / No | A / B / C / ... | 1,234,567 | Several groups of photos |
| Methods | Logistic regression, Support Vector Machine | KNN | Regression | K-means, Topic Modeling |
| Sample Dataset | `breast cancer` | `iris` | `diabetes` | TBD | 

This tutorial consists of three parts: 1) Data Preparation, 2) Supervised Machine Learning, 3) Unsupervised Machine Learning, and 4) Evaluation of Machine Learning Model. `scikit-learn` is used for example source codes in this tutorial for simplicity and clarity. 

1. [Sample datasets for this tutorial](Dataset.md)
   - breast cancer
   - iris
   - diabetes

2. [Data preparation](Data_preparation.md)
   - Reading raw data
   - Inspecting dimension(rows, columns) and data type
   - Inspecting value of the dataset
   - Visualizing correlation among columns

3. [Supervised machine learning](Supervised.md)
   - Regression
     - Linear regression
   - Binary classification
     - Logistic regression
     - Support vector machine
   - Multiclass classification
     - K-nearest neighbors
     - Decision tree
     - Random forests
   - Overfitting
     - K fold cross validation
   - Hyperparameter optimization
     - Gridsearch
4. [Unsupervised machine learning](Unsupervised.md)
   - K-means clustering
   - Topic modeling
5. [Evaluation of machine learning model](Evaluation.md)
   - Confusion matrix
   - Accuracy, precision, recall
   - Receiver Operating Charasteristic(ROC) curve: Trus positive rate(TPR), False positive rate(FPR)
   - Area Under the ROC Curve(AUC)
6. [Source code template](Source_code.md)
   - Supervised model
   - Unsupervised model
   - Reusability

### Miscellaneous

- Major class : The class with larger number of samples than the other in the dataset.
- Minor class : The class with less number of samples than the other in the dataset.
- Baseline accuracy : The proportion of the major class in the dataset.

#
# Data Preparation
#
from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#
# Bread cancer
#

#
# Read dataset from library
#
raw_dataset = datasets.load_breast_cancer()

# Create pandas dataframe
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target
print(raw_dataset.target_names)
print(my_dataset['class'].value_counts())
#
# Check data structure, # of rows, # of columns
#
print('\nData structure\n')
my_dataset.info()

#
# Read dataset from library
#
raw_dataset = datasets.load_iris()

# Create pandas dataframe
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target
print(raw_dataset.target_names)
print(my_dataset['class'].value_counts())
#
# Check data structure, # of rows, # of columns
#
print('\nData structure\n')
my_dataset.info()

#
# Diabetes
#
raw_dataset = datasets.load_diabetes()

# Create pandas dataframe
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target
print(my_dataset['class'].value_counts())
#
# Check data structure, # of rows, # of columns
#
print('\nData structure\n')
my_dataset.info()

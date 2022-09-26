#
# Neural network using PyTorch
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
import math

###############################################################################
#
# Define neural network model
#
###############################################################################
class CustomDataset(Dataset):
    def __init__(self,raw_dataset):
        df = raw_dataset
        self.x = df.loc[:, df.columns !='class'].values
        self.y = df.loc[:,'class'].values
        self.length = len(df)
    def __getitem__(self, index):
        x = torch.FloatTensor(self.x[index,:])
        y = torch.FloatTensor(self.y[index])
        return x, y
    def __len__(self):
        return self.length

class MyNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MyNetwork, self).__init__()
        self.layer = nn.Linear(input_dim,1)
    def forward(self,x):
        x = self.layer(x)
        return x

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

#
# Read dataset from library
#
raw_dataset = datasets.load_breast_cancer()
my_dataset = pd.DataFrame(raw_dataset.data, columns=raw_dataset.feature_names)
my_dataset['class'] = raw_dataset.target

#
# Convert numeric value to string in class
#
# my_dataset['class'].replace([0,1],raw_dataset.target_names, inplace=True)

#
# Check data structure, # of rows, # of columns
#
print('\nData structure\n')
my_dataset.info()

#
# Check value of data
#
print('\nData value\n')
print(my_dataset.head())

#
# Count data per class
#
print('\nCount by class\n')
print(my_dataset['class'].value_counts())

#
# Descriptive statistics
#
print('\nDescriptive statistics\n')
print(my_dataset.describe())

# X = my_dataset.drop('class',axis=1)
# y = my_dataset['class']

dataset    = CustomDataset(my_dataset)
train_size = math.floor(len(dataset) * 0.8)
test_size  = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset,[train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
# print(f'\nTraining set : {X_train.shape}, Test set : {y_train.shape}\n')
# print(f'\nTraining set : {train_dataset.shape}, Test set : {test_dataset.shape}\n')

clf = MyNetwork(30)
criterion = nn.MSELoss()
optimizer = optim.SGD(clf.parameters(), lr=0.0001)

###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')
for epoch in range(10):
    cost = 0.0
    for X, y in train_dataloader:
        output = clf(X)
        loss = criterion(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    cost = cost/len(train_dataloader)
    if (epoch+1) % 10 == 0:
        print(f'Epoch: {epoch}, Cost: {cost}')

###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\nPredicting...\n')

with torch.no_grad():
    clf.eval()
    for X,y in test_dataloader:
        output = clf(X.float())
        print(f'y: {y}')
        print(f'Output: {output}')
        print(f'----------------')
        

###############################################################################
#
# 4. Evaluating
#
###############################################################################
# print('\Evaluating...\n')

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# disp.plot()
# plt.show()

# print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
# print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
# print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
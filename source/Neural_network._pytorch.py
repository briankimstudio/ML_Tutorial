#
# Neural network using PyTorch
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
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
class MyNetwork(nn.Module):
    def __init__(self, input_dim):
        super(MyNetwork, self).__init__()
        self.layer = nn.Sequential (
            nn.Linear(input_dim,60),
            nn.Linear(60,1),
            nn.Sigmoid()
        )
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
print(my_dataset.describe().T)


X = my_dataset.drop('class',axis=1)
y = my_dataset['class']
y = y.to_frame()

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X))

print('\nDescriptive statistics after standardization\n')
print(X.describe().T)

# train_size = math.floor(len(my_dataset) * 0.8)
# test_size  = len(my_dataset) - train_size

# train_dataset, test_dataset = random_split(dataset,[train_size, test_size])

#
# Split dataset into traning(X_train, y_train) set and test set(X_test, y_test).
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

X_train = torch.FloatTensor(X_train.values)
y_train = torch.FloatTensor(y_train.values)
X_test = torch.FloatTensor(X_test.values)
y_test = torch.FloatTensor(y_test.values)

print(f'\nTraining set : {X_train.shape}, Test set : {y_train.shape}\n')
# print(f'\nTraining set : {train_dataset.shape}, Test set : {test_dataset.shape}\n')

clf = MyNetwork(30)
# criterion = nn.MSELoss(size_average=False)
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(clf.parameters(), lr=0.001)
epochs = 10000
###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')
losses = []
for epoch in range(epochs):
    clf.train()
    output = clf(X_train)
    loss = criterion(output,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 100 == 0:
        losses.append(loss.detach().numpy())
        print(f'Epoch: {epoch:6d}, Loss: {loss:.20f}, AUC: {metrics.roc_auc_score(y_train, output.detach().numpy()):.20f}')

###############################################################################
#
# 3. Estimating
#
###############################################################################
# print('\nPredicting...\n')

clf.eval()
with torch.no_grad():
    y_pred = clf(X_test)
    cm = confusion_matrix(y_test.int().squeeze(), torch.where(y_pred.squeeze()>0.5,1,0))
    print(cm)
    print(f'\nAUC score : {metrics.roc_auc_score(y_test, y_pred)}')
#     for X,y in test_dataloader:
#         output = clf(X.float())
#         print(f'y: {y}')
#         print(f'Output: {output}')
#         print(f'----------------')
        

###############################################################################
#
# 4. Evaluating
#
###############################################################################
plt.plot(losses)
plt.title('Loss vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# print('\Evaluating...\n')

# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
# disp.plot()
# plt.show()

# print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
# print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
# print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
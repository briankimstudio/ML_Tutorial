
#
# Unsupervised Machine Learning Template
#

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

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

print(f'\nData structure\n')
print(my_dataset.info())

print(f'\nData preview\n')
print(my_dataset.head())

###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining Model...\n')

###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\Estimating Model...\n')

###############################################################################
#
# 4. Evaluating
#
###############################################################################
print('\Evaluating Results...\n')
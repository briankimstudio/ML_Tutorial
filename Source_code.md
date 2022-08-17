## Source code structure

In this tutorial, most of python source codes follow the same structure as below.

![template](/images/template.png)

```
#
# Supervised Machine Learning Template
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

###############################################################################
#
# 2. Training
#
###############################################################################
print('\nTraining...\n')

###############################################################################
#
# 3. Estimating
#
###############################################################################
print('\nPredicting...\n')

###############################################################################
#
# 4. Evaluating
#
###############################################################################
print('\Evaluating...\n')
```

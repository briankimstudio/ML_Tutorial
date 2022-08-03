
#
# Title
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

###############################################################################
#
# 2. Training
#
###############################################################################

###############################################################################
#
# 3. Estimating
#
###############################################################################

###############################################################################
#
# 4. Evaluating
#
###############################################################################

cm = confusion_matrix(y_test, y_pred)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = clf.classes_)
disp.plot()
plt.show()

print(f'\nAccuracy  : {metrics.accuracy_score(y_test,y_pred)}')
print(f'Precision : {metrics.precision_score(y_test, y_pred)}')
print(f'Recall    : {metrics.recall_score(y_test, y_pred)}')
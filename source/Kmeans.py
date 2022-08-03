#
# K Means Clustering
#

from sklearn.cluster import KMeans
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
X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

plt.title('Data')
plt.scatter(x=X[:,0], y=X[:,1])
plt.show()

print(X)
###############################################################################
#
# 2. Training
#
###############################################################################
cluster_num = 2
clf = KMeans(n_clusters = cluster_num, random_state=0)
clf.fit(X)

print(clf.labels_)

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

print(clf.cluster_centers_)

#
# K Means Clustering
#

from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#
# To generate consistant experimental results
#
# np.random.seed(0)

###############################################################################
#
# 1. Data preparation
#
###############################################################################
raw_dataset = datasets.make_blobs(n_samples=200,n_features=2, centers=7,
                                  cluster_std=1, random_state=0 )

# plt.scatter(raw_dataset[0][:,0],raw_dataset[0][:,1],c=raw_dataset[1],cmap='rainbow')
# plt.xlabel('x1')
# plt.xlabel('x2')
# plt.show()
# X = np.array([[1, 2], [1, 4], [1, 0],
#               [10, 2], [10, 4], [10, 0]])

# plt.title('Data')
# plt.scatter(x=X[:,0], y=X[:,1])
# plt.show()

# print(X)
wcss = []
for i in range(1,11):
    clf = KMeans(n_clusters=i,init='k-means++', random_state=150)
    clf.fit(raw_dataset[0])
    wcss.append(clf.inertia_)
plt.plot(range(1,11),wcss, color='k')
plt.xlabel('number of clusters')
plt.ylabel('WCSS')
plt.show()


###############################################################################
#
# 2. Training
#
###############################################################################
# cluster_num = 2
# clf = KMeans(n_clusters = cluster_num, random_state=0)
# clf.fit(X)

# print(clf.labels_)

clf = KMeans(n_clusters=5, init='k-means++')
clf.fit(raw_dataset[0])

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

f, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(raw_dataset[0][:,0],raw_dataset[0][:,1],c=clf.labels_,cmap='rainbow')
ax1.set_title('Original')
ax2.scatter(raw_dataset[0][:,0],raw_dataset[0][:,1],c=raw_dataset[1],cmap='rainbow')
plt.show()
# print(clf.cluster_centers_)

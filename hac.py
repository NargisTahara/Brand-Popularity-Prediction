# needed imports
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
# some setting for this notebook to actually show the graphs inline, you probably won't need this
from dataset import Datasets

np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation
# generate two clusters: a with 100 points, b with 50:
ds = Datasets()
X = ds.getAllCont()
print(X.shape)  # 150 samples with 2 dimensions

# generate the linkage matrix
Z = linkage(X, method = 'ward')
c, coph_dists = cophenet(Z, pdist(X))

plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
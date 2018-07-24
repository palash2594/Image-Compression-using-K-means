
from skimage import io
from sklearn.cluster import KMeans
import numpy as np

image = io.imread('scenery.png')
io.imshow(image)
io.show()


rows = image.shape[0]
cols = image.shape[1]
hi = image.shape[2]

print(rows, cols)
clusters = int(input("Enter number of clusters"))

image = (image / 255.0).reshape(rows * cols, 3)
kmeans = KMeans(n_clusters = clusters)
kmeans.fit(image)

compressed_image = kmeans.cluster_centers_[kmeans.labels_]

compressed_image = compressed_image.reshape(rows, cols, hi);

io.imsave('compressed_scenery.png', compressed_image)
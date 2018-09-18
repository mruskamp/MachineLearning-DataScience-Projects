import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib.factorization.python.ops import kmeans

def input_fn_1D(input_1D_):
    input_t = tf.convert_to_tensor(input_1D_, dtype=tf.float32)
    input_t = tf.expand_dims(input_t, 1)
    return(input_t, None)

input_1D = np.array([1,2,3.0,4,5,126,21,33,6,73.0,2,3,56,98,100,4,8,33,102])

k_means_estimator = kmeans.KMeansClustering(num_clusters=2)
fit = k_means_estimator.train(input_fn=lambda: input_fn_1D(input_1D), steps=1000)
clusters_1D = k_means_estimator.cluster_centers()

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(input_1D, np.zeros_like(input_1D), s=300, marker='o')
ax1.scatter(clusters_1D, np.zeros_like(clusters_1D), c='r', s=200, marker='s')
plt.show()

for var in fit.get_variable_names():
    print(var, "-----> ", fit.get_variable_value(var))














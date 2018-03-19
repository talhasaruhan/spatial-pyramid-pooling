## TEST

from spp_layer import np_spatial_pyramid_pooling, tf_spatial_pyramid_pooling
import numpy as np
import tensorflow as tf

a = np.random.randint(100, size=(1, 3, 6, 5)).astype(np.int32)

spt = np.array([
[3, 5], [2, 2]
])

print(a.shape)
print(a)

fixed_size_representation = np_spatial_pyramid_pooling(a, spt, dtype=np.int32)
print(fixed_size_representation)

with tf.Session() as sess:
	tf_a = tf.constant(a)
	tf_spt = tf.constant(spt)
	y = tf_spatial_pyramid_pooling(tf_a, tf_spt, tf.int32)
	print(sess.run(y))
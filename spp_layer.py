import numpy as np
import math
import tensorflow as tf

# input feature maps is of the form: C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def spatial_pyramid_pooling(input_feature_maps, spatial_pyramid):
	assert spatial_pyramid.shape[1] == 2

	num_levels = spatial_pyramid.shape[0]
	bins_per_level = np.prod(spatial_pyramid, axis=1)
	num_bins = np.sum(bins_per_level)
	fixed_length_representation = np.zeros((num_bins), dtype=np.float32)

	num_channels = input_feature_maps.shape[0]
	# C-W*H
	flattened_feature_maps = np.reshape(input_feature_maps, (num_channels, -1))
	# stride tricks, then max pool along one dimension 
	# then stride tricks again and max pool along the other dimension
	# but whats the length and stride?
	# the original paper uses the method I first thought of
	# ceil(w/n) for window size, floor(w/n) for stride,
	# where w is the original dim, and n is the number of bins along the dim 
	# but this implementation may leave out some pixels (consider w = 5, n = 3)
	sizeof_int32 = np.dtype(np.int32).itemsize
	sizeof_float32 = np.dtype(np.float32).itemsize

	n = 3
	w = input_feature_maps.shape[2]
	h = input_feature_maps.shape[1]
	num_px = flattened_feature_maps.shape[1]
	l = math.ceil(w/n)
	s = math.floor(w/n)
	# d = math.floor((w-l)/s)+1
	ar = np.lib.stride_tricks.as_strided(flattened_feature_maps, (num_channels, h, n, l), 
		(sizeof_int32*num_px, sizeof_int32*w, sizeof_int32*s, sizeof_int32))
	# print(num_channels, w, n, s, d, l)
	# print(ar.shape)
	# print(ar)
	ar = np.amax(ar, axis=3)
	# print(ar.shape)
	# print(ar)
	ar = np.transpose(ar, (0, 2, 1)).copy()
	# print(ar.shape)
	# print(ar)	
	l = math.ceil(h/n)
	s = math.floor(h/n)
	ar = np.lib.stride_tricks.as_strided(ar, (num_channels, n, n, l), 
		(sizeof_int32*n*h, sizeof_int32*h, sizeof_int32*s, sizeof_int32))
	# print(num_channels, w, n, s, d, l)
	# print(ar.shape)
	# print(ar)
	ar = np.transpose(np.amax(ar, axis=3), (0, 2, 1))

	print(ar.shape)
	print(ar)

	pass

a = np.random.randint(10, size=(3, 6, 5))

spt = np.array([
[1, 1], [2, 2]
])

print(a.shape)
print(a)
spatial_pyramid_pooling(a, spt)
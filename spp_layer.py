import numpy as np
import math
import tensorflow as tf

# input feature maps is of the form: C-(WH)/(HW)
# ex. spatial_pyramid:
#	[[1, 1], [2, 2], [3, 3], [4, 5]]
# each row is a level of pyramid with nxm pooling
def spatial_pyramid_pooling(input_feature_maps, spatial_pyramid):
	np_spatial_pyramid = np.array(spatial_pyramid)
	assert np_spatial_pyramid.shape[1] == 2

	w = input_feature_maps.shape[2]
	h = input_feature_maps.shape[1]

	num_levels = np_spatial_pyramid.shape[0]
	num_channels = input_feature_maps.shape[0]
	# C-W*H
	flattened_feature_maps = np.reshape(input_feature_maps, (num_channels, -1))
	num_px = flattened_feature_maps.shape[1]

	bins_per_level = np.prod(np_spatial_pyramid, axis=1)
	num_bins = np.sum(bins_per_level)
	stack = []

	# stride tricks, then max pool along one dimension 
	# then stride tricks again and max pool along the other dimension
	# but whats the length and stride?
	# the original paper uses the method I first thought of
	# ceil(w/n) for window size, floor(w/n) for stride,
	# where w is the original dim, and n is the number of bins along the dim 
	# but this implementation may leave out some pixels (consider w = 5, n = 3)
	sizeof_int32 = np.dtype(np.int32).itemsize
	sizeof_float32 = np.dtype(np.float32).itemsize

	for n_h, n_w in spatial_pyramid:
		l = math.ceil(w/n_w)
		s = math.floor(w/n_w)
		# d = math.floor((w-l)/s)+1
		ar = np.lib.stride_tricks.as_strided(flattened_feature_maps, (num_channels, h, n_w, l), 
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
		l = math.ceil(h/n_h)
		s = math.floor(h/n_h)
		ar = np.lib.stride_tricks.as_strided(ar, (num_channels, n_w, n_h, l), 
			(sizeof_int32*n_w*h, sizeof_int32*h, sizeof_int32*s, sizeof_int32))
		# print(num_channels, w, n, s, d, l)
		# print(ar.shape)
		# print(ar)
		ar = np.transpose(np.amax(ar, axis=3), (0, 2, 1))
		# print(ar.shape)
		# print(ar)
		stack.append(np.reshape(ar, (num_channels, -1)))
		# print(stack[-1].shape)

	stack = np.concatenate(stack, axis=1)
	print(stack.shape)

	return stack

a = np.random.randint(10, size=(3, 6, 5))

spt = [
(3, 5), (2, 2)
]

print(a.shape)
print(a)
fixed_size_representation = spatial_pyramid_pooling(a, spt)
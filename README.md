# spatial-pyramid-pooling
Spatial Pyramid Pooling layer implemented in NumPy and Tensorflow

For the original SPP paper: arXiv:1406.4729 [cs.CV]

* **spp_layer.py** includes functions:
  * ***np_spatial_pyramid_pooling(input_feature_maps, pyramid_levels, dtype)***
    * **input_feature_maps** : <br /> Numpy array of 4 dims, following NCHW format.
    * **pyramid_levels** : <br /> Numpy array of 2 dims. <br /> Rows represent each level of the pyramid. First column represent number of bins along H dimension (n_H) and the second column represent number of bins along W dimension (n_W).
    * **dtype** <br/> Data type of the numpy array (i.e. np.int32). This has to be set correctly as this function uses "stride tricks" and rely on memory continuity.
  * ***tf_spatial_pyramid_pooling(input_feature_maps, pyramid_levels, dtype)***
    * **input_feature_maps**: <br/> Tensor of 4 dims, following NCHW format.
    * **pyramid_levels**: <br/> Tensor of 2 dims.
    * **dtype**: <br/> Data type of the tensor (i.e. tf.int32).
* **test.py** includes a simple test bench for the functions

**Note**: The flattened fixed-size representation will preserve the order between pooling levels. I.E. If you have a *3X5* level, then a *2x2* one, first 15 entries of the resulting array will correspond to flattened representation of the output of pooling operation.

An example:

Create a random array of (N, C, H, W), and let the pyramid levels be ((3, 5), (2, 2))
```python
a = np.random.randint(100, size=(1, 3, 6, 5)).astype(np.int32)

spt = np.array([
[3, 5], [2, 2]
])

print(a.shape)
print(a)
```

Generated array:
```
(1, 3, 6, 5)
[[[[70 40 51 25 50]
   [78 20 94 43 46]
   [22 69 97 72 83]
   [32 90 32 47 90]
   [38 65 94 66 88]
   [24 10 32 15 91]]

  [[74 96 83 55 76]
   [76 60 37 80 36]
   [52  4 55 56 18]
   [78 49 42 55 94]
   [55 23 51 33 78]
   [47 60 10 95 36]]

  [[65  0  8 58 50]
   [42 49 23  4 54]
   [30 81 53  2 55]
   [81 27 97 73 25]
   [67 41 53 21  3]
   [83 10 33 79 97]]]]
```

Numpy implementation:
```python
fixed_size_representation = np_spatial_pyramid_pooling(a, spt, dtype=np.int32)
print(fixed_size_representation.shape)
print(fixed_size_representation)
```

Output:
```
(1, 3, 19)
[[[78 40 94 43 50 32 90 97 72 90 38 65 94 66 91 97 97 94 94]
  [76 96 83 80 76 78 49 55 56 94 55 60 51 95 78 96 83 78 95]
  [65 49 23 58 54 81 81 97 73 55 83 41 53 79 97 81 58 97 97]]]
```

Tensorflow implementation:
```python
with tf.Session() as sess:
	tf_a = tf.constant(a)
	tf_spt = tf.constant(spt)
	y = tf_spatial_pyramid_pooling(tf_a, tf_spt, tf.int32)
	tf_fxd_repr = sess.run(y)
	print(tf_fxd_repr.shape)
	print(tf_fxd_repr)
```

Output:
```
(1, 3, 19)
[[[78 40 94 43 50 32 90 97 72 90 38 65 94 66 91 97 97 94 94]
  [76 96 83 80 76 78 49 55 56 94 55 60 51 95 78 96 83 78 95]
  [65 49 23 58 54 81 81 97 73 55 83 41 53 79 97 81 58 97 97]]]
```

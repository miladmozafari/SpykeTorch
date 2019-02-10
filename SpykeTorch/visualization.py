import torch
from   PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Show 2D the tensor.
def show_tensor(aTensor, _vmin = None, _vmax = None):
	r"""Plots a 2D tensor in gray color map and shows it in a window.

	Args:
		aTensor (Tensor): The input tensor.
		_vmin (float, optional): Minimum value. Default: None
		_vmax (float, optional): Maximum value. Default: None

	.. note::

		:attr:`None` for :attr:`_vmin` or :attr:`_vmin` causes an auto-scale mode for each.
	"""
	if aTensor.is_cuda:
		aTensor = aTensor.cpu()
	plt.figure()
	plt.imshow(aTensor.numpy(),cmap='gray', vmin=_vmin, vmax=_vmax)
	plt.colorbar()
	plt.show()

def plot_tensor_in_image(fname, aTensor, _vmin = None, _vmax = None):
	r"""Plots a 2D tensor in gray color map in an image file.

	Args:
		fname (str): The file name.
		aTensor (Tensor): The input tensor.
		_vmin (float, optional): Minimum value. Default: None
		_vmax (float, optional): Maximum value. Default: None

	.. note::

		:attr:`None` for :attr:`_vmin` or :attr:`_vmin` causes an auto-scale mode for each.
	"""
	if aTensor.is_cuda:
		aTensor = aTensor.cpu()
	plt.imsave(fname,aTensor.numpy(),cmap='gray', vmin=_vmin, vmax=_vmax)

# Computes window size of a neuron on specific previous layer
# layer_details must be a sequence of quaraples containing (height, width, row_stride, col_stride)
# of each layer
def get_deep_receptive_field(*layers_details):
	h,w = 1,1
	for height,width,r_stride,c_stride in reversed(layers_details):
		h = height + (h-1) * r_stride
		w = width + (w-1) * c_stride
	return h,w

# Computes the feature that a neuron is selective to given the feature of the neurons underneath
# The cumulative stride (which is cumulative product of previous layers' strides) must be given
# The stride of the previous layer must be given
# pre_feature is the 3D tensor of the features for underlying neurons
# feature_stride is the cumulative stride (tuple) = (height, width)
# stride is the stride of previous layer (tuple) = (height, width)
# weights is the 4D weight tensor of current layer (None if it is a pooling layer)
# retruns features and the new cumulative stride
def get_deep_feature(pre_feature, feature_stride, window_size, stride, weights=None):
	new_cstride = (feature_stride[0] * stride[0], feature_stride[1] * stride[1])
	new_size = (pre_feature.size(-2) + (window_size[0]-1) * feature_stride[0],
			 pre_feature.size(-1) + (window_size[1]-1) * feature_stride[1])
	depth = pre_feature.size(-3)
	if weights is not None:
		depth = weights.size(0)
	new_feature = torch.zeros(depth, *new_size, device=pre_feature.device)
	if weights is None: # place the feature in the middle of the field
		start_point = (new_size[0]//2-pre_feature.size(-2)//2,new_size[1]//2-pre_feature.size(-1)//2)
		new_feature[:,start_point[0]:start_point[0]+pre_feature.size(-2),start_point[1]:start_point[1]+pre_feature.size(-1)] = pre_feature
	else:
		for r in range(weights.size(-2)): #rows
			for c in range(weights.size(-1)): #cols
				temp_features = pre_feature * weights[:,:,r:r+1,c:c+1]
				temp_features = temp_features.max(dim=1)[0]
				new_feature[:,r*feature_stride[0]:r*feature_stride[0]+pre_feature.size(-2),
				c*feature_stride[1]:c*feature_stride[1]+pre_feature.size(-1)] += temp_features
		new_feature.clamp_(min=0) # removing negatives

	return new_feature,new_cstride
import torch
import torch.nn.functional as fn
import numpy as np
import math
from   PIL import Image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def read_image(address, scaling_factor = 1, _device=None):
	img = Image.open(address,'r')
	if(scaling_factor != 1):
		img = resize_image(img, scaling_factor)
	img = img.convert('L') #makes it greyscale
	img = np.asarray(img.getdata(),dtype=np.float32).reshape((img.size[1],img.size[0]))
	res = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
	if _device == 'cuda':
		res = res.cuda()
	return res

# resizing the image while keeping the aspect ratio
# img is a PIL image
# factor is a real value where the base is 1
def resize_image(img, factor):
	new_size = (int(img.size[0]*factor), int(img.size[1]*factor))
	return img.resize(new_size, Image.ANTIALIAS)

# packs DoG parameters
class DoGParam:
	def __init__(self, window_size, sigma1, sigma2):
		self.window_size = window_size
		self.sigma1 = sigma1
		self.sigma2 = sigma2

# packs Gabor parameters
class GaborParam:
	def __init__(self, window_size, orientation):
		self.window_size = window_size
		self.orientation = orientation

# returns a 2d tensor corresponding to the requested DoG filter
def get_DoG_kernel_manual(window_size, sigma1, sigma2):
	w = window_size//2
	x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
	a = 1.0 / (2 * math.pi)
	prod = x*x + y*y
	f1 = (1/(sigma1*sigma1)) * np.exp(-0.5 * (1/(sigma1*sigma1)) * (prod))
	f2 = (1/(sigma2*sigma2)) * np.exp(-0.5 * (1/(sigma2*sigma2)) * (prod))
	dog = a * (f1-f2)
	dog_mean = np.mean(dog)
	dog = dog - dog_mean
	dog_max = np.max(dog)
	dog = dog / dog_max
	dog_tensor = torch.from_numpy(dog)
	return dog_tensor.float()

# returns a 2d tensor corresponding to the requested Gabor filter
def get_gabor_kernel_manual(window_size, orientation, div=4.0):
	w = window_size//2
	x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
	lamda = window_size * 2 / div
	sigma = lamda * 0.8
	sigmaSq = sigma * sigma
	g = 0.3;
	theta = (orientation * np.pi) / 180;
	Y = y*np.cos(theta) - x*np.sin(theta)
	X = y*np.sin(theta) + x*np.cos(theta)
	gabor = np.exp(-(X * X + g * g * Y * Y) / (2 * sigmaSq)) * np.cos(2 * np.pi * X / lamda);
	gabor_mean = np.mean(gabor)
	gabor = gabor - gabor_mean
	gabor_max = np.max(gabor)
	gabor = gabor / gabor_max
	gabor_tensor = torch.from_numpy(gabor)
	return gabor_tensor.float()

# returns a 4d tensor of all input DoG filters
# dimensions (dog_filters, groups = 1, height, width)
# DoG_paramaeters is a list of DoGParam
def get_multi_DoG_kernel(DoG_parameters,_device=None):
	tensor_list = []
	max_window_size = 0
	for param in DoG_parameters:
		max_window_size = max(max_window_size, param.window_size)
	for param in DoG_parameters:
		p = (max_window_size - param.window_size)//2
		tensor_list.append(fn.pad(get_DoG_kernel_manual(param.window_size, param.sigma1, param.sigma2).unsqueeze(0), (p,p,p,p)))
	final = torch.stack(tensor_list)
	if _device == 'cuda':
		final = final.cuda()
	return final

# returns a 4d tensor of all input Gabor filters
# dimensions (gabor_filters, groups = 1, height, width)
# gabor_paramaeters is a list of GaborParam
def get_multi_gabor_kernel(gabor_parameters,_device=None):
	tensor_list = []
	max_window_size = 0
	for param in gabor_parameters:
		max_window_size = max(max_window_size, param.window_size)
	for param in gabor_parameters:
		p = (max_window_size - param.window_size)//2
		tensor_list.append(fn.pad(get_gabor_kernel_manual(param.window_size, param.orientation).unsqueeze(0), (p,p,p,p)))
	final = torch.stack(tensor_list)
	if _device == 'cuda':
		final = final.cuda()
	return final
	
# returns a 4d tensor containing the flitered versions of the input image
# input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
def apply_filters(input_img, multi_filter_kernel, threshold=0, use_abs=False):
	output = fn.conv2d(input_img, multi_filter_kernel).float()
	if use_abs:
		torch.abs_(output)
	fn.threshold_(output, threshold, 0)
	return output

def apply_filters2(input_img, multi_filter_kernel, thresholds, use_abs=False):
	output = fn.conv2d(input_img, multi_filter_kernel).float()
	if use_abs:
		torch.abs_(output)
	#fn.threshold_(output, threshold, 0)
	return torch.where(output < thresholds, torch.tensor(0.0, device=output.device), output)

# performs local normalization
# on each region (of size radius*2 + 1) the mean value is computed and 
# intensities will be divided by the mean value
# x is a 4D tensor
def local_normalization(x, radius, eps=1e-12):
	# computing local mean by 2d convolution
	kernel = torch.ones(1,1,radius*2+1,radius*2+1,device=x.device).float()/((radius*2+1)**2)
	# rearrange 4D tensor so input channels will be considered as minibatches
	y = x.squeeze(0) # removes minibatch dim which was 1
	y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
	means = fn.conv2d(y,kernel,padding=radius) + eps # computes means
	y = y/means # normalization
	# swap minibatch with channels
	y.squeeze_(1)
	y.unsqueeze_(0)
	return y

# returns a tensor of tensors containing spikes in each timestep (considers minibatch for timesteps)
# spikes are accumulative, i.e. spikes in timestep i are also presented in i+1, i+2, ...
def intensity_to_latency(intencities, timesteps):
	bins = []
	bins_intencities = []
	nonzero_cnt = torch.nonzero(intencities).size()[0]
	#check for empty bins
	bin_size = nonzero_cnt//timesteps

	#sort
	intencities_flattened = torch.reshape(intencities, (-1,))
	intencities_flattened_sorted = torch.sort(intencities_flattened, descending=True)
	#print(intencities_flattened)
	#bin packing
	sorted_bins_value, sorted_bins_idx = torch.split(intencities_flattened_sorted[0], bin_size), torch.split(intencities_flattened_sorted[1], bin_size)
	#print(sorted_bins_idx)
	#add to the list of timesteps
	spike_map = torch.zeros_like(intencities_flattened_sorted[0])
	
	for i in range(timesteps):
		spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
		spike_map_copy = torch.tensor(spike_map)
		spike_map_copy = spike_map_copy.reshape(tuple(intencities.shape))
		bins_intencities.append(spike_map_copy.squeeze(0).float())
		bins.append(spike_map_copy.sign().squeeze_(0).float())
	
	return torch.stack(bins_intencities),torch.stack(bins)

# decrease lateral intencities by factors given in the inhibition_kernel
def intensity_lateral_inhibition(intencities, inhibition_kernel):
	inh_win_size = inhibition_kernel.size(-1)
	rad = inh_win_size//2
	# repeat each value
	values = intencities.reshape(intencities.size(0),intencities.size(1),-1,1)
	values = values.repeat(1,1,1,inh_win_size)
	values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
	values = values.repeat(1,1,1,inh_win_size)
	values = values.reshape(intencities.size(0),intencities.size(1),-1,intencities.size(-1)*inh_win_size)
	# extend patches
	padded = fn.pad(intencities,(rad,rad,rad,rad))
	# column-wise
	patches = padded.unfold(-1,inh_win_size,1)
	patches = patches.reshape(patches.size(0),patches.size(1),patches.size(2),-1,patches.size(3)*patches.size(4))
	patches.squeeze_(-2)
	# row-wise
	patches = patches.unfold(-2,inh_win_size,1).transpose(-1,-2)
	patches = patches.reshape(patches.size(0),patches.size(1),1,-1,patches.size(-1))
	patches.squeeze_(-3)
	# compare each element by its neighbors
	coef = values - patches
	coef.clamp_(min=0).sign_() # "ones" are neighbors greater than center
	# convolution with full stride to get accumulative inhibiiton factor
	factors = fn.conv2d(coef, inhibition_kernel, stride=inh_win_size)

	## rescaling to [0,1]
	#min_v = factors.min()
	#range_v = factors.max() - min_v
	#if range_v > 0:
	#	factors = (factors.sub(other=min_v)) / range_v
	#elif min_v > 0:
	#	factors = torch.ones_like(factors)
	#else:
	#	factors = torch.zeros_like(factors)

	result = intencities + intencities * factors
	#result = (result - result.min()) + 1
	return result

# full means to do zero padding for full convolution
def convolution(inputs, weights, full=False):
	if full:
		return fn.conv2d(inputs, weights, padding=(weights.size(-2)//2,weights.size(-1)//2))
	else:
		return fn.conv2d(inputs, weights)

# max-pooling
# window and stride are tuples (height,width), and (rows,cols)
def pooling(inputs, window, stride, full=False):
	if full:
		return fn.max_pool2d(inputs,window,stride,padding=(window.size(-2)//2,window.size(-1)//2))
	else:
		return fn.max_pool2d(inputs,window,stride)

# padding
def pad(x, pad, value=0):
	return fn.pad(x,pad,value=value)

def threshold(potentials, threshold):
	outputs = torch.tensor(potentials)
	if threshold == 'Inf':
		outputs[:-1]=0
	else:
		fn.threshold_(outputs, threshold, 0)
	return outputs

def fire(potentials):
	return potentials.sign()

def threshold_(potentials, threshold):
	if threshold=='Inf':
		potentials[:-1]=0
	else:
		fn.threshold_(potentials, threshold, 0)

def fire_(potentials):
	potentials.sign_()

# in each position, the most fitted feature will survive (first earliest spike then maximum potential)
# it is assumed that the threshold function is applied on the input potentials
def pointwise_inhibition(potentials):
	# maximum of each position in each time step
	maximum = torch.max(potentials, dim=1, keepdim=True)
	# detection of the earliest spike
	clamp_pot = maximum[0].sign()
	# maximum of clamped values is the indices of the earliest spikes
	clamp_pot_max = torch.max(clamp_pot, dim=0, keepdim=True)
	# finding winners (maximum potentials between early spikes)
	winners = maximum[1].gather(0,clamp_pot_max[1])
	# generating inhibition coefficient
	coef = torch.zeros_like(potentials[0]).unsqueeze_(0)
	coef.scatter_(1,winners,clamp_pot_max[0])
	# applying inhibition to potentials (broadcasting multiplication)
	return torch.mul(potentials,coef)

# each position inhibits its srrounding if it is higher than all of them
# it is assumed that the threshold function is applied on the input potentials
def featurewise_lateral_inhibition(potentials,inhibition_radius):
	spikes = potentials.sign()
	# finding earliest potentials for each position in each feature
	maximum = torch.max(spikes, dim=0, keepdim=True) # finding earliest index
	values = potentials.gather(dim=0, index=maximum[1]) # gathering values
	# propagating the earliest potential through the whole timesteps
	truncated_pot = spikes * values
	# summation with a high enough value (maximum of potential summation over timesteps) at spike positions
	total = truncated_pot.sum(dim=0,keepdim=True)
	v = total.max()
	truncated_pot.addcmul_(spikes,v)
	# summation over all timesteps
	total = truncated_pot.sum(dim=0,keepdim=True)
	# max pooling
	pool_val,pool_idx = fn.max_pool2d(total, 2*inhibition_radius+1, inhibition_radius+1, return_indices = True)
	# generating inhibition kernel
	total = fn.max_unpool2d(input=pool_val, indices=pool_idx, kernel_size=2*inhibition_radius+1, stride=inhibition_radius+1,output_size=total.size()).sign_()
	# applyong inhibition
	return torch.mul(potentials,total)

# inhibiting particular features, preventing them to be winners
# inhibited_features is a list of features numbers to be inhibited
def feature_inhibition_(potentials,inhibited_features):
	if len(inhibited_features) != 0:
		potentials[:,inhibited_features,:,:] = 0

# returns list of winners and inhibition coefficient (to be applied on the original potentials if needed)
# inhibition_radius is to increase the chance of diversity among features (if needed)
# it is assumed that the threshold function is applied on the input potentials
# winners_values contains pairs of (time, potential) of winners
def get_k_winners(potentials, kwta, inhibition_radius=0):
	k = 0
	t = 0
	# making a copy of original potentials inorder to apply temporal inhibitions
	potentials_copy = torch.tensor(potentials)
	# finding global pooling size
	global_pooling_size = potentials[0].size()
	stride = (1,1,1)
	coef = torch.ones_like(potentials[0])
	winners = []
	winners_values = []
	while t != potentials.size(0) and k != kwta:
		# reformating potentials of a single timestep to a 5d tensor acceptable for pool3d
		current = potentials_copy[t].unsqueeze(0).unsqueeze_(0)
		# finding the global maximum
		max_val,max_idx = fn.max_pool3d(current, global_pooling_size, stride, return_indices = True)
		# if there is spike
		if max_val != 0:
			# finding the 3d position of the maximum value
			max_idx_unraveled = np.unravel_index(max_idx.item(),global_pooling_size)
			# adding to the winners list
			winners.append(max_idx_unraveled)
			winners_values.append((t, max_val))
			# preventing the same feature to be the next winner
			index_tensor = torch.from_numpy(np.array([max_idx_unraveled[0]]))
			if potentials.is_cuda:
				index_tensor = index_tensor.cuda()
			
			coef.index_fill_(0,index_tensor,0)
			# columnar inhibition (increasing the chance of leanring diverse features)
			if inhibition_radius != 0:
				rowMin,rowMax = max(0,max_idx_unraveled[1]-inhibition_radius),min(coef.size(1),max_idx_unraveled[1]+inhibition_radius+1)
				colMin,colMax = max(0,max_idx_unraveled[2]-inhibition_radius),min(coef.size(2),max_idx_unraveled[2]+inhibition_radius+1)
				coef[:,rowMin:rowMax,colMin:colMax] = 0
			
			potentials_copy = torch.mul(potentials_copy,coef)
			k+=1
		else:
			t+=1
	return winners, coef, winners_values

# each position inhibits its srrounding if it is higher than all of them
# it is assumed that the threshold function is applied on the input potentials
def get_k_winners2(potentials,kwta,inhibition_radius):
	spikes = potentials.sign()
	# finding earliest potentials for each position in each feature
	maximum = torch.max(spikes, dim=0, keepdim=True) # finding earliest index
	values = potentials.gather(dim=0, index=maximum[1]) # gathering values
	# propagating the earliest potential through the whole timesteps
	truncated_pot = spikes * values
	# summation with a high enough value (maximum of potential summation over timesteps) at spike positions
	#total = truncated_pot.sum(dim=0,keepdim=True)
	#v = total.max()
	v = truncated_pot.max() * potentials.size(0)
	truncated_pot.addcmul_(spikes,v)
	# summation over all timesteps
	total = truncated_pot.sum(dim=0,keepdim=True)
	
	winners = []
	# adding dummy dimension for max_pool3d
	total.squeeze_(0)
	global_pooling_size = tuple(total.size())#[-3:]
	stride = (1,1,1)
	for k in range(kwta):
		#max_val,max_idx = fn.max_pool3d(total, global_pooling_size, stride, return_indices = True)
		max_val,max_idx = total.view(-1).max(0)
		if max_val.item() != 0:
			# finding the 3d position of the maximum value
			max_idx_unraveled = np.unravel_index(max_idx.item(),global_pooling_size)
			# adding to the winners list
			winners.append(max_idx_unraveled)
			# preventing the same feature to be the next winner
			total[max_idx_unraveled[0],:,:] = 0
			# columnar inhibition (increasing the chance of leanring diverse features)
			if inhibition_radius != 0:
				rowMin,rowMax = max(0,max_idx_unraveled[-2]-inhibition_radius),min(total.size(-2),max_idx_unraveled[-2]+inhibition_radius+1)
				colMin,colMax = max(0,max_idx_unraveled[-1]-inhibition_radius),min(total.size(-1),max_idx_unraveled[-1]+inhibition_radius+1)
				total[:,rowMin:rowMax,colMin:colMax] = 0
		else:
			break
	return winners

# returns a list of byte tensors corresponding to winners, each element represents pre-post(1), post-pre(0)
# input and outputs spikes must be binary tensors
# receptive_filed_size is a ternary tuple (depth,height,width)
# winners are ternary tuples indicating the position of the winner (depth,row,column)
# assuming that there is no zero padding for convolution
def get_pre_post_ordering(input_spikes, output_spikes, winners, receptive_field_size):
	# accumulating input and output spikes to get latencies
	input_latencies = torch.sum(input_spikes, dim=0)
	output_latencies = torch.sum(output_spikes, dim=0)
	result = []
	for winner in winners:
		# generating repeated output tensor with the same size of the receptive field
		out_tensor = torch.ones(*receptive_field_size,device=output_latencies.device) * output_latencies[winner]
		# slicing input tensor with the same size of the receptive field centered around winner
		# since there is no padding, there is no need to shift it to the center
		in_tensor = input_latencies[:,winner[-2]:winner[-2]+receptive_field_size[-2],winner[-1]:winner[-1]+receptive_field_size[-1]]
		result.append(torch.ge(in_tensor,out_tensor))
	return result

# simple STDP rule
# gets prepost pairings, winners, weights, and learning rates (all shoud be tensors)
def STDP(weights, pairings, winners, ap, an):
	for i in range(len(winners)):
		lr = torch.where(pairings[i], ap, an)
		weights[winners[i][0]] += lr * weights[winners[i][0]] * (1-weights[winners[i][0]])

# simple STDP rule
# gets prepost pairings, winners, weights, and learning rates (all shoud be tensors)
# faster (needs to be check to be the same)
def STDP2(weights, pairings, winners, ap, an):
	lr = torch.zeros_like(weights)
	for i in range(len(winners)):
		lr[winners[i][0]] = torch.where(pairings[i], ap, an)
	weights += lr * weights * (1-weights)
	weights.clamp_(0,1)

# simple STDP rule with bounds without stabilizer
# gets prepost pairings, winners, weights, and learning rates (all shoud be tensors except bounds)
# faster (needs to be check to be the same)
def STDP2_bounded(weights, pairings, winners, ap, an, lb, ub):
	lr = torch.zeros_like(weights)
	for i in range(len(winners)):
		lr[winners[i][0]] = torch.where(pairings[i], ap, an)
	weights += lr
	weights.clamp_(lb,ub)

# simple STDP rule
# gets prepost pairings, winners, weights, and learning rates (all shoud be tensors and list of tensors)
def STDP_multirate(weights, pairings, winners, aps, ans):
	for i in range(len(winners)):
		lr = torch.where(pairings[i], aps[winners[i][0]], ans[winners[i][0]])
		weights[winners[i][0]] += lr * weights[winners[i][0]] * (1-weights[winners[i][0]])

# Show 2D the tensor.
def show_tensor(aTensor, _vmin = None, _vmax = None):
	if aTensor.is_cuda:
		aTensor = aTensor.cpu()
	plt.figure()
	plt.imshow(aTensor.numpy(),cmap='gray', vmin=_vmin, vmax=_vmax)
	plt.colorbar()
	plt.show()

def plot_tensor_in_image(fname, aTensor, _vmin = None, _vmax = None):
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
	#new_size = (pre_feature.size(-2) + (window_size[0]-1) * new_cstride[0],
	#		 pre_feature.size(-1) + (window_size[1]-1) * new_cstride[1])
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
		# loop over synapses in different positions
		for r in range(weights.size(-2)): #rows
			for c in range(weights.size(-1)): #cols
				temp_features = pre_feature * weights[:,:,r:r+1,c:c+1]
				temp_features = temp_features.max(dim=1)[0]
				new_feature[:,r*feature_stride[0]:r*feature_stride[0]+pre_feature.size(-2),
				c*feature_stride[1]:c*feature_stride[1]+pre_feature.size(-1)] += temp_features
		new_feature.clamp_(min=0) # removing negatives

	return new_feature,new_cstride


#############################
#MAIN
#############################
def main():
	print('this is main')

if __name__== "__main__":
	main()

import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
from .utils import to_pair

# padding
# pad = (padLeft, padRight, padTop, padBottom)
def pad(x, pad, value=0):
	return fn.pad(x, pad, value=value)

# pooling
def pooling(input, kernel_size, stride=None, padding=0):
	return fn.max_pool2d(input, kernel_size, stride, padding)

# if return_thresholded_potentials is True, thresholded potentials will be returned as the second value
# None for threshold means infinite threshold
def fire(potentials, threshold=None, return_thresholded_potentials=False):
	thresholded = torch.tensor(potentials)
	if threshold is None:
		thresholded[:-1]=0
	else:
		fn.threshold_(thresholded, threshold, 0)
	if return_thresholded_potentials:
		return thresholded.sign(), thresholded
	return thresholded.sign()

def fire_(potentials, threshold=None):
	if threshold is None:
		potentials[:-1]=0
	else:
		fn.threshold_(potentials, threshold, 0)
	potentials.sign_()

def threshold(potentials, threshold=None):
	outputs = torch.tensor(potentials)
	if threshold is None:
		outputs[:-1]=0
	else:
		fn.threshold_(outputs, threshold, 0)
	return outputs

def threshold_(potentials, threshold=None):
	if threshold is None:
		potentials[:-1]=0
	else:
		fn.threshold_(potentials, threshold, 0)

# in each position, the most fitted feature will survive (first earliest spike then maximum potential)
# it is assumed that the threshold function is applied on the input potentials
def pointwise_inhibition(thresholded_potentials):
	# maximum of each position in each time step
	maximum = torch.max(thresholded_potentials, dim=1, keepdim=True)
	# compute signs for detection of the earliest spike
	clamp_pot = maximum[0].sign()
	# maximum of clamped values is the indices of the earliest spikes
	clamp_pot_max = torch.max(clamp_pot, dim=0, keepdim=True)
	# finding winners (maximum potentials between early spikes)
	winners = maximum[1].gather(0, clamp_pot_max[1])
	# generating inhibition coefficient
	coef = torch.zeros_like(thresholded_potentials[0]).unsqueeze_(0)
	coef.scatter_(1, winners,clamp_pot_max[0])
	# applying inhibition to potentials (broadcasting multiplication)
	return torch.mul(thresholded_potentials, coef)

# each position inhibits its srrounding if it is higher than all of them
# it is assumed that the threshold function is applied on the input potentials
def featurewise_lateral_inhibition(potentials, inhibition_radius, spikes=None):
	if spikes is None:
		spikes = potentials.sign()
	# finding earliest potentials for each position in each feature
	maximum = torch.max(spikes, dim=0, keepdim=True) # finding earliest index
	values = potentials.gather(dim=0, index=maximum[1]) # gathering values
	# propagating the earliest potential through the whole timesteps
	truncated_pot = spikes * values
	# summation with a high enough value (maximum of potential summation over timesteps) at spike positions
	total = truncated_pot.sum(dim=0, keepdim=True)
	v = total.max()
	truncated_pot.addcmul_(spikes,v)
	# summation over all timesteps
	total = truncated_pot.sum(dim=0,keepdim=True)
	# max pooling
	pool_val,pool_idx = fn.max_pool2d(total, 2*inhibition_radius+1, inhibition_radius+1, return_indices = True)
	# generating inhibition kernel
	#total = fn.max_unpool2d(input=pool_val, indices=pool_idx, kernel_size=2*inhibition_radius+1, stride=inhibition_radius+1,output_size=total.size()).clamp_(0,1)
	total = fn.max_unpool2d(input=pool_val, indices=pool_idx, kernel_size=2*inhibition_radius+1, stride=inhibition_radius+1,output_size=total.size()).sign_()
	# applyong inhibition
	return torch.mul(potentials, total)

# inhibiting particular features, preventing them to be winners
# inhibited_features is a list of features numbers to be inhibited
def feature_inhibition_(potentials, inhibited_features):
	if len(inhibited_features) != 0:
		potentials[:, inhibited_features, :, :] = 0

def feature_inhibition(potentials, inhibited_features):
	potentials_copy = torch.tensor(potentials)
	if len(inhibited_features) != 0:
		feature_inhibition_(potentials_copy, inhibited_features)
	return potentials_copy

# returns list of winners
# inhibition_radius is to increase the chance of diversity among features (if needed)
def get_k_winners(potentials, kwta = 1, inhibition_radius = 0, spikes = None):
	if spikes is None:
		spikes = potentials.sign()
	# finding earliest potentials for each position in each feature
	maximum = torch.max(spikes, dim=0, keepdim=True) # finding earliest index
	values = potentials.gather(dim=0, index=maximum[1]) # gathering values
	# propagating the earliest potential through the whole timesteps
	truncated_pot = spikes * values
	# summation with a high enough value (maximum of potential summation over timesteps) at spike positions
	v = truncated_pot.max() * potentials.size(0)
	truncated_pot.addcmul_(spikes,v)
	# summation over all timesteps
	total = truncated_pot.sum(dim=0,keepdim=True)
	
	total.squeeze_(0)
	global_pooling_size = tuple(total.size())
	winners = []
	for k in range(kwta):
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

# decrease lateral intencities by factors given in the inhibition_kernel
def intensity_lateral_inhibition(intencities, inhibition_kernel):
	intencities.squeeze_(0)
	intencities.unsqueeze_(1)

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
	result = intencities + intencities * factors

	intencities.squeeze_(1)
	intencities.unsqueeze_(0)
	result.squeeze_(1)
	result.unsqueeze_(0)
	return result

# performs local normalization
# on each region (of size radius*2 + 1) the mean value is computed and 
# intensities will be divided by the mean value
# x is a 4D tensor
def local_normalization(x, normalization_radius, eps=1e-12):
	# computing local mean by 2d convolution
	kernel = torch.ones(1,1,normalization_radius*2+1,normalization_radius*2+1,device=x.device).float()/((normalization_radius*2+1)**2)
	# rearrange 4D tensor so input channels will be considered as minibatches
	y = x.squeeze(0) # removes minibatch dim which was 1
	y.unsqueeze_(1)  # adds a dimension after channels so previous channels are now minibatches
	means = fn.conv2d(y,kernel,padding=normalization_radius) + eps # computes means
	y = y/means # normalization
	# swap minibatch with channels
	y.squeeze_(1)
	y.unsqueeze_(0)
	return y

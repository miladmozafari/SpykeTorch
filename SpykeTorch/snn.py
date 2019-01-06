import torch
import torch.nn as nn
import torch.nn.functional as fn
from . import functional as sf
from torch.nn.parameter import Parameter
from .utils import to_pair

class Convolution(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding=0, weight_mean=0.8, weight_std=0.02):
		super(Convolution, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = to_pair(kernel_size)
		self.padding = to_pair(padding)
		self.weight_mean = weight_mean
		self.weight_std = weight_std

		# For future use
		self.stride = 1
		self.bias = None
		self.dilation = 1
		self.groups = 1

		# Parameters
		self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
		self.weight.requires_grad_(False) # We do not use gradients
		self.reset_weight()

	def reset_weight(self):
		self.weight.normal_(self.weight_mean, self.weight_std)

	def forward(self, input):
		return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Pooling(nn.Module):
	def __init__(self, kernel_size, stride=None, padding=0):
		super(Pooling, self).__init__()
		self.kernel_size = to_pair(kernel_size)
		if stride is None:
			self.stride = self.kernel_size
		else:
			self.stride = self.kernel_size
		self.padding = to_pair(padding)

		# For future use
		self.dilation = 1
		self.return_indices = False
		self.ceil_mode = False

	def forward(self, input):
		return sf.pooling(input, self.kernel_size, self.stride, self.padding)

class STDP(nn.Module):
	def __init__(self, conv_layer, learning_rate, use_stabilizer = True, lower_bound = 0, upper_bound = 1):
		super(STDP, self).__init__()
		self.conv_layer = conv_layer
		if isinstance(learning_rate, list):
			self.learning_rate = learning_rate
		else:
			self.learning_rate = [learning_rate] * conv_layer.out_channels
		for i in range(conv_layer.out_channels):
			#self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]], device=conv_layer.weight.device)),
			#				Parameter(torch.tensor([self.learning_rate[i][1]], device=conv_layer.weight.device)))
			self.learning_rate[i] = (Parameter(torch.tensor([self.learning_rate[i][0]])),
							Parameter(torch.tensor([self.learning_rate[i][1]])))
			self.register_parameter('ltp_' + str(i), self.learning_rate[i][0])
			self.register_parameter('ltd_' + str(i), self.learning_rate[i][1])
			self.learning_rate[i][0].requires_grad_(False)
			self.learning_rate[i][1].requires_grad_(False)
		self.use_stabilizer = use_stabilizer
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

	# returns a list of byte tensors corresponding to winners, each element represents pre-post(1), post-pre(0)
	# input and outputs spikes must be binary tensors
	# receptive_filed_size is a ternary tuple (depth,height,width)
	# winners are ternary tuples indicating the position of the winner (depth,row,column)
	def get_pre_post_ordering(self, input_spikes, output_spikes, winners):
		# accumulating input and output spikes to get latencies
		input_latencies = torch.sum(input_spikes, dim=0)
		output_latencies = torch.sum(output_spikes, dim=0)
		result = []
		for winner in winners:
			# generating repeated output tensor with the same size of the receptive field
			out_tensor = torch.ones(*self.conv_layer.kernel_size, device=output_latencies.device) * output_latencies[winner]
			# slicing input tensor with the same size of the receptive field centered around winner
			# since there is no padding, there is no need to shift it to the center
			in_tensor = input_latencies[:,winner[-2]:winner[-2]+self.conv_layer.kernel_size[-2],winner[-1]:winner[-1]+self.conv_layer.kernel_size[-1]]
			result.append(torch.ge(in_tensor,out_tensor))
		return result

	# simple STDP rule
	# gets prepost pairings, winners, weights, and learning rates (all shoud be tensors)
	def forward(self, input_spikes, potentials, output_spikes, winners=None, kwta = 1, inhibition_radius = 0):
		if winners is None:
			winners = sf.get_k_winners(potentials, kwta, inhibition_radius, output_spikes)
		pairings = self.get_pre_post_ordering(input_spikes, output_spikes, winners)
		
		lr = torch.zeros_like(self.conv_layer.weight)
		for i in range(len(winners)):
			f = winners[i][0]
			lr[f] = torch.where(pairings[i], *(self.learning_rate[f]))

		self.conv_layer.weight += lr * (self.conv_layer.weight * (1-self.conv_layer.weight) if self.use_stabilizer else 1)
		self.conv_layer.weight.clamp_(self.lower_bound, self.upper_bound)

	def update_learning_rate(self, feature, ap, an):
		self.learning_rate[feature][0][0] = ap
		self.learning_rate[feature][1][0] = an

	def update_all_learning_rate(self, ap, an):
		for feature in range(self.conv_layer.out_channels):
			self.learning_rate[feature][0][0] = ap
			self.learning_rate[feature][1][0] = an
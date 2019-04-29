###################################################################################
# Reimplementation of the Digit Recognition Experiment (MNIST) Performed in:      #
# https://www.sciencedirect.com/science/article/pii/S0893608017302903             #
#                                                                                 #
# Reference:                                                                      #
# Kheradpisheh, Saeed Reza, et al.                                                #
# "STDP-based spiking deep convolutional neural networks for object recognition." #
# Neural Networks 99 (2018): 56-67.                                               #
#                                                                                 #
###################################################################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms
import struct
import glob

use_cuda = True

class KheradpishehMNIST(nn.Module):
	def __init__(self):
		super(KheradpishehMNIST, self).__init__()

		self.conv1 = snn.Convolution(2, 32, 5, 0.8, 0.05)
		self.conv1_t = 10
		self.k1 = 5
		self.r1 = 2

		self.conv2 = snn.Convolution(32, 150, 2, 0.8, 0.05)
		self.conv2_t = 1
		self.k2 = 8
		self.r2 = 1

		self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
		self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
		self.max_ap = Parameter(torch.Tensor([0.15]))

		self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
		self.spk_cnt1 = 0
		self.spk_cnt2 = 0
	
	def save_data(self, input_spike, potentials, output_spikes, winners):
		self.ctx["input_spikes"] = input_spike
		self.ctx["potentials"] = potentials
		self.ctx["output_spikes"] = output_spikes
		self.ctx["winners"] = winners

	def forward(self, input, max_layer):
		input = sf.pad(input, (2,2,2,2), 0)
		if self.training:
			pot = self.conv1(input)
			spk, pot = sf.fire(pot, self.conv1_t, True)
			if max_layer == 1:
				self.spk_cnt1 += 1
				if self.spk_cnt1 >= 500:
					self.spk_cnt1 = 0
					ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
					ap = torch.min(ap, self.max_ap)
					an = ap * -0.75
					self.stdp1.update_all_learning_rate(ap.item(), an.item())
				pot = sf.pointwise_inhibition(pot)
				spk = pot.sign()
				winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
				self.save_data(input, pot, spk, winners)
				return spk, pot
			spk_in = sf.pad(sf.pooling(spk, 2, 2, 1), (1,1,1,1))
			spk_in = sf.pointwise_inhibition(spk_in)
			pot = self.conv2(spk_in)
			spk, pot = sf.fire(pot, self.conv2_t, True)
			if max_layer == 2:
				pot = sf.pointwise_inhibition(pot)
				spk = pot.sign()
				winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
				self.save_data(spk_in, pot, spk, winners)
				return spk, pot
			spk_out = sf.pooling(spk, 2, 2, 1)
			return spk_out
		else:
			pot = self.conv1(input)
			spk, pot = sf.fire(pot, self.conv1_t, True)
			pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2, 1), (1,1,1,1)))
			spk, pot = sf.fire(pot, self.conv2_t, True)
			spk = sf.pooling(spk, 2, 2, 1)
			return spk
	
	def stdp(self, layer_idx):
		if layer_idx == 1:
			self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
		if layer_idx == 2:
			self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
	network.train()
	for i in range(len(data)):
		data_in = data[i]
		if use_cuda:
			data_in = data_in.cuda()
		network(data_in, layer_idx)
		network.stdp(layer_idx)

def test(network, data, target, layer_idx):
	network.eval()
	ans = [None] * len(data)
	t = [None] * len(data)
	for i in range(len(data)):
		data_in = data[i]
		if use_cuda:
			data_in = data_in.cuda()
		output,_ = network(data_in, layer_idx).max(dim = 0)
		ans[i] = output.reshape(-1).cpu().numpy()
		t[i] = target[i]
	return np.array(ans), np.array(t)

class S1Transform:
	def __init__(self, filter, timesteps = 15):
		self.to_tensor = transforms.ToTensor()
		self.filter = filter
		self.temporal_transform = utils.Intensity2Latency(timesteps)
		self.cnt = 0
	def __call__(self, image):
		if self.cnt % 1000 == 0:
			print(self.cnt)
		self.cnt+=1
		image = self.to_tensor(image) * 255
		image.unsqueeze_(0)
		image = self.filter(image)
		image = sf.local_normalization(image, 8)
		temporal_image = self.temporal_transform(image)
		return temporal_image.sign()

kernels = [ utils.DoGKernel(7,1,2),
			utils.DoGKernel(7,2,1),]
filter = utils.Filter(kernels, padding = 3, thresholds = 50)
s1 = S1Transform(filter)

data_root = "data"
MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1))
MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1))
MNIST_loader = DataLoader(MNIST_train, batch_size=len(MNIST_train), shuffle=False)
MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

kheradpisheh = KheradpishehMNIST()
if use_cuda:
	kheradpisheh.cuda()

# Training The First Layer
print("Training the first layer")
if os.path.isfile("saved_l1.net"):
	kheradpisheh.load_state_dict(torch.load("saved_l1.net"))
else:
	for epoch in range(2):
		print("Epoch", epoch)
		iter = 0
		for data,_ in MNIST_loader:
			print("Iteration", iter)
			train_unsupervise(kheradpisheh, data, 1)
			print("Done!")
			iter+=1
	torch.save(kheradpisheh.state_dict(), "saved_l1.net")

# Training The Second Layer
print("Training the second layer")
if os.path.isfile("saved_l2.net"):
	kheradpisheh.load_state_dict(torch.load("saved_l2.net"))
for epoch in range(20):
	print("Epoch", epoch)
	iter = 0
	for data,_ in MNIST_loader:
		print("Iteration", iter)
		train_unsupervise(kheradpisheh, data, 2)
		print("Done!")
		iter+=1
torch.save(kheradpisheh.state_dict(), "saved_l2.net")

# Classification
# Get train data
for data,target in MNIST_loader:
	train_X, train_y = test(kheradpisheh, data, target, 2)
	

# Get test data
for data,target in MNIST_testLoader:
	test_X, test_y = test(kheradpisheh, data, target, 2)

# SVM
from sklearn.svm import LinearSVC
clf = LinearSVC(C=2.4)
clf.fit(train_X, train_y)
predict_train = clf.predict(train_X)
predict_test = clf.predict(test_X)

def get_performance(X, y, predictions):
	correct = 0
	silence = 0
	for i in range(len(predictions)):
		if X[i].sum() == 0:
			silence += 1
		else:
			if predictions[i] == y[i]:
				correct += 1
	return (correct/len(X), (len(X)-(correct+silence))/len(X), silence/len(X))

print(get_performance(train_X, train_y, predict_train))
print(get_performance(test_X, test_y, predict_test))

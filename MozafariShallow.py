##########################################################################
# Reimplementation of the Object Recognition Experiments Performed in:   #
# https://ieeexplore.ieee.org/document/8356226/                          #
#                                                                        #
# Reference:                                                             #
# Mozafari, Milad, et al.,                                               #
# "First-Spike-Based Visual Categorization Using Reward-Modulated STDP.",#
# IEEE Transactions on Neural Networks and Learning Systems (2018).      #
#                                                                        #
# Original Implementation (in C#):                                       #
# https://senselab.med.yale.edu/ModelDB/showmodel.cshtml?model=240369    #
##########################################################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

class Mozafari2018(nn.Module):
    def __init__(self, input_channels, features_per_class, number_of_classes,
              s2_kernel_size, threshold, stdp_lr, anti_stdp_lr, dropout = 0.):
        super(Mozafari2018, self).__init__()
        self.features_per_class = features_per_class
        self.number_of_classes = number_of_classes
        self.number_of_features = features_per_class * number_of_classes
        self.kernel_size = s2_kernel_size
        self.threshold = threshold
        self.stdp_lr = stdp_lr
        self.anti_stdp_lr = anti_stdp_lr
        self.dropout = torch.ones(self.number_of_features) * dropout
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

        self.s2 = snn.Convolution(input_channels, self.number_of_features, self.kernel_size, 0.8, 0.05)
        self.stdp = snn.STDP(self.s2, stdp_lr)
        self.anti_stdp = snn.STDP(self.s2, anti_stdp_lr)
        self.decision_map = []
        for i in range(number_of_classes):
            self.decision_map.extend([i]*features_per_class)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
    
    def forward(self, input):
        input = input.float()
        pot = self.s2(input)
        
        if self.training and self.dropout[0] > 0:
            sf.feature_inhibition_(pot, self.to_be_dropped)

        spk, pot = sf.fire(pot, self.threshold, True)
        winners = sf.get_k_winners(pot, 1, 0, spk)
        output = -1
        if len(winners) != 0:
            output = self.decision_map[winners[0][0]]

        if self.training:
            self.ctx["input_spikes"] = input
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
        else:
            self.ctx["input_spikes"] = None
            self.ctx["potentials"] = None
            self.ctx["output_spikes"] = None
            self.ctx["winners"] = None

        return output

    def update_dropout(self):
        self.to_be_dropped = torch.bernoulli(self.dropout).nonzero()

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

class S1C1Transform:
    def __init__(self, filter, pooling_size, pooling_stride, lateral_inhibition = None, timesteps = 15,
              feature_wise_inhibition=True):
        self.grayscale = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.lateral_inhibition = lateral_inhibition
        self.temporal_transform = utils.Intensity2Latency(timesteps)
        self.feature_wise_inhibition = feature_wise_inhibition

    def __call__(self, image):
        image = self.to_tensor(self.grayscale(image))
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.pooling(image, self.pooling_size, self.pooling_stride, padding=self.pooling_size//2)
        if self.lateral_inhibition is not None:
            image = self.lateral_inhibition(image)
        temporal_image = self.temporal_transform(image)
        temporal_image = sf.pointwise_inhibition(temporal_image)
        return temporal_image.sign().byte()

kernels = [	utils.GaborKernel(5, 45+22.5),
            utils.GaborKernel(5, 90+22.5),
            utils.GaborKernel(5, 135+22.5),
            utils.GaborKernel(5, 180+22.5)]

filter = utils.Filter(kernels, use_abs = True)
lateral_inhibition = utils.LateralIntencityInhibition([0.15, 0.12, 0.1, 0.07, 0.05])

task = "Caltech"
use_cuda = True

if task == "Caltech":
    s1c1 = S1C1Transform(filter, 7, 6, lateral_inhibition)
    trainsetfolder = utils.CacheDataset(ImageFolder("facemotortrain", s1c1))
    testsetfolder = utils.CacheDataset(ImageFolder("facemotortest", s1c1))
    mozafari = Mozafari2018(4, 10, 2, (17,17), 42, (0.005, -0.0025), (-0.005, 0.0005), 0.5)
    trainset = DataLoader(trainsetfolder, batch_size = len(trainsetfolder), shuffle = True)
    testset = DataLoader(testsetfolder, batch_size = len(testsetfolder), shuffle = True)
    max_epoch = 400
elif task == "ETH":
    s1c1 = S1C1Transform(filter, 5, 4, lateral_inhibition)
    mozafari = Mozafari2018(4, 10, 8, (31,31), 160, (0.01, -0.0035), (-0.01, 0.0006), 0.4)

    def target_transform(target):
        return target//10
    datafolder = utils.CacheDataset(ImageFolder("eth80-cropped-close128", s1c1, target_transform=target_transform))
    test_instances = np.random.randint(0, 10, 8)
    train_indices = set(range(len(datafolder)))
    test_indices = set()
    for c in range(8):
        for i in range(41):
            test_indices.add(c * 410 + test_instances[c] * 41 + i)
    train_indices -= test_indices
    train_indices = list(train_indices)
    test_indices = list(test_indices)
    trainset = DataLoader(datafolder, batch_size = 8 * 9 * 41, sampler=torch.utils.data.SubsetRandomSampler(train_indices))
    testset = DataLoader(datafolder, batch_size = 8 * 1 * 41, sampler=torch.utils.data.SubsetRandomSampler(test_indices))
    max_epoch = 250
elif task == "Norb":
    s1c1 = S1C1Transform(filter, 5, 4, lateral_inhibition, timesteps=30)
    trainsetfolder = utils.CacheDataset(ImageFolder("norb/train", s1c1))
    testsetfolder = utils.CacheDataset(ImageFolder("norb/test", s1c1))
    mozafari = Mozafari2018(4, 10, 5, (23,23), 150, (0.05, -0.003), (-0.05, 0.0005), 0.5)
    trainset = DataLoader(trainsetfolder, batch_size = len(trainsetfolder), shuffle = True)
    testset = DataLoader(testsetfolder, batch_size = len(testsetfolder), shuffle = True)
    max_epoch = 800

if use_cuda:
    mozafari.cuda()


# initial adaptive learning rates
apr = mozafari.stdp_lr[0]
anr = mozafari.stdp_lr[1]
app = mozafari.anti_stdp_lr[1]
anp = mozafari.anti_stdp_lr[0]

adaptive_min = 0.2
adaptive_int = 0.8
apr_adapt = ((1.0 - 1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / mozafari.number_of_classes) * adaptive_int + adaptive_min) * anp

# perf
best_train = np.array([0,0,0,0]) # correct, wrong, silence, epoch
best_test = np.array([0,0,0,0]) # correct, wrong, silence, epoch

# train one batch (here a batch contains all data so it is an epoch)
def train(data, target, network):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    network.update_dropout()
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in)
        if d != -1:
            if d == target_in:
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)

# test one batch (here a batch contains all data so it is an epoch)
def test(data, target, network):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in)
        if d != -1:
            if d == target_in:
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)

for epoch in range(max_epoch):
    print("Epoch #:", epoch)
    for data, target in trainset:
        perf_train = train(data, target, mozafari)
    if best_train[0] <= perf_train[0]:
        best_train = np.append(perf_train, epoch)
    print("Current Train:", perf_train)
    print("   Best Train:", best_train)
    for data_test, target_test in testset:
        perf_test = test(data_test, target_test, mozafari)
    if best_test[0] <= perf_test[0]:
        best_test = np.append(perf_test, epoch)
        torch.save(mozafari.state_dict(), "saved.net")
    print(" Current Test:", perf_test)
    print("    Best Test:", best_test)

    #update adaptive learning rates
    apr_adapt = apr * (perf_train[1] * adaptive_int + adaptive_min)
    anr_adapt = anr * (perf_train[1] * adaptive_int + adaptive_min)
    app_adapt = app * (perf_train[0] * adaptive_int + adaptive_min)
    anp_adapt = anp * (perf_train[0] * adaptive_int + adaptive_min)
    mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
    
# Features #
feature = torch.tensor([
    [
        [1]
    ]
    ]).float()
if use_cuda:
    feature = feature.cuda()

cstride = (1,1)

# S1 Features #
if use_cuda:
    feature,cstride = vis.get_deep_feature(feature, cstride, (filter.max_window_size, filter.max_window_size), (1,1), filter.kernels.cuda())
else:
    feature,cstride = vis.get_deep_feature(feature, cstride, (filter.max_window_size, filter.max_window_size), (1,1), filter.kernels)
# C1 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, (s1c1.pooling_size, s1c1.pooling_size), (s1c1.pooling_stride, s1c1.pooling_stride))
# S2 Features #
feature,cstride = vis.get_deep_feature(feature, cstride, mozafari.kernel_size, (1,1), mozafari.s2.weight)

for i in range(mozafari.number_of_features):
    vis.plot_tensor_in_image('feature_s2_'+str(i).zfill(4)+'.png',feature[i])

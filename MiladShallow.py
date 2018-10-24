import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
import SpykeTorch as st
import numpy as np
import glob
import math

# Random Seed
np.random.seed(100)
torch.manual_seed(100)

# GPU or CPU
_dvc = 'cuda'

# Task
task = 'caltech'

# Train network or not?
need_training = True

# Dataset #
if task == 'caltech':
	dataset = []
	dataset.append(np.array(glob.glob('/home/milad_mozafari/research/Python/caltech/all/face/*png')))
	dataset.append(np.array(glob.glob('/home/milad_mozafari/research/Python/caltech/all/motor/*png')))

	# Train-Test Split #
	train_percent = 0.5
	training_set = []
	training_lbl = []
	testing_set = []
	testing_lbl = []
	for c in range(len(dataset)):
		perm = np.random.permutation(len(dataset[c]))
		cnt = int(train_percent * len(perm))
		training_set.append(dataset[c][perm[0:cnt]])
		training_lbl.append(np.repeat(c,cnt))
		testing_set.append(dataset[c][perm[cnt:]])
		testing_lbl.append(np.repeat(c,len(dataset[c])-cnt))
	training_set = np.concatenate(training_set)
	training_lbl = np.concatenate(training_lbl)
	testing_set = np.concatenate(testing_set)
	testing_lbl = np.concatenate(testing_lbl)

if task == 'eth':
	prefix = "/home/milad_mozafari/research/Python/eth/"
	postfix = "/*png"
	prefix = "C:\\Me\\PhD\\Datasets\\eth80-cropped-close128\\all\\"
	postfix = "\\*png"
	dataset = []
	dataset.append(np.array(glob.glob(prefix + 'Apple' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Car' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Cow' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Cup' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Dog' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Horse' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Pear' + postfix)))
	dataset.append(np.array(glob.glob(prefix + 'Tomato' + postfix)))
	class_names = np.array(['apple','car','cow','cup','dog','horse','pear','tomato'])
	
	# Train-Test Split #
	test_instances = np.random.randint(1, 11, len(dataset))
	training_set = []
	training_lbl = []
	testing_set = []
	testing_lbl = []
	for c in range(len(dataset)):
		test_str = str(class_names[c])+str(test_instances[c])+'-'
		for add in dataset[c]:
			if test_str in add:
				testing_set.append(add)
				testing_lbl.append(c)
			else:
				training_set.append(add)
				training_lbl.append(c)

if task == 'norb':
	prefix = "/home/milad_mozafari/research/Python/norb/train/"
	postfix = "/*png"
	prefix = "C:\\Me\\PhD\\Datasets\\NORB\\train\\"
	postfix = "\\*png"
	dataset = []
	dataset.append(np.array(glob.glob(prefix + '0' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '1' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '2' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '3' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '4' + postfix)))
	
	# Train-Test Split #
	training_set = []
	training_lbl = []
	for c in range(len(dataset)):
		for add in dataset[c]:
			training_set.append(add)
			training_lbl.append(c)

	prefix = "/home/milad_mozafari/research/Python/norb/test/"
	postfix = "/*png"
	prefix = "C:\\Me\\PhD\\Datasets\\NORB\\test\\"
	postfix = "\\*png"
	dataset = []
	dataset.append(np.array(glob.glob(prefix + '0' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '1' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '2' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '3' + postfix)))
	dataset.append(np.array(glob.glob(prefix + '4' + postfix)))

	testing_set = []
	testing_lbl = []
	for c in range(len(dataset)):
		for add in dataset[c]:
			testing_set.append(add)
			testing_lbl.append(c)

# Gabor Kernels #
gabor = st.get_multi_gabor_kernel([st.GaborParam(5,45+22.5),
								   st.GaborParam(5,90+22.5),
								   st.GaborParam(5,135+22.5),
								   st.GaborParam(5,180+22.5)])
if _dvc == 'cuda':
	gabor = gabor.cuda()

# Features #
feature = torch.tensor([
	[
		[1]
	]
	]).float()

cstride = (1,1)
if _dvc == 'cuda':
	feature = feature.cuda()

if task == 'caltech':
	c1_pooling_win = (7,7)
	c1_pooling_stride = (6,6)
	number_of_features = 20
	neuron_per_class = number_of_features//len(dataset)
	receptive_field = 17
	kwta = 1
	threhsold = 42
	dropout = 0.5
	p_drop = torch.ones(number_of_features) * dropout
	adaptive_min = 0.2
	adaptive_int = 0.8
	neuron_labels = []
	for c in range(len(dataset)):
		neuron_labels += [c]*neuron_per_class

	apr = torch.tensor(0.005,device=_dvc)
	anr = torch.tensor(-0.0025,device=_dvc)
	app = torch.tensor(0.0005,device=_dvc)
	anp = torch.tensor(-0.005,device=_dvc)

	scale_factor  = 0.5
	timesteps = 15

if task == 'eth':
	c1_pooling_win = (5,5)
	c1_pooling_stride = (4,4)
	number_of_features = 80
	neuron_per_class = number_of_features//len(dataset)
	receptive_field = 31
	kwta = 1
	threhsold = 160
	dropout = 0.4
	p_drop = torch.ones(number_of_features) * dropout
	adaptive_min = 0.2
	adaptive_int = 0.8
	neuron_labels = []
	for c in range(len(dataset)):
		neuron_labels += [c]*neuron_per_class

	apr = torch.tensor(0.01,device=_dvc)
	anr = torch.tensor(-0.0035,device=_dvc)
	app = torch.tensor(0.0006,device=_dvc)
	anp = torch.tensor(-0.01,device=_dvc)

	scale_factor  = 1
	timesteps = 15

if task == 'norb':
	c1_pooling_win = (5,5)
	c1_pooling_stride = (4,4)
	number_of_features = 50
	neuron_per_class = number_of_features//len(dataset)
	receptive_field = 23
	kwta = 1
	threhsold = 150
	dropout = 0.5
	p_drop = torch.ones(number_of_features) * dropout
	adaptive_min = 0.2
	adaptive_int = 0.8
	neuron_labels = []
	for c in range(len(dataset)):
		neuron_labels += [c]*neuron_per_class

	apr = torch.tensor(0.05,device=_dvc)
	anr = torch.tensor(-0.003,device=_dvc)
	app = torch.tensor(0.0005,device=_dvc)
	anp = torch.tensor(-0.05,device=_dvc)

	scale_factor  = 1
	timesteps = 100

# inhibition kernel
inhibition_kernel = torch.tensor([
	[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
	[0.05, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.12, 0.15, 0.15, 0.15, 0.12, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.12, 0.15, 0.00, 0.15, 0.12, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.12, 0.15, 0.15, 0.15, 0.12, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.12, 0.12, 0.12, 0.12, 0.12, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.07, 0.05],
	[0.05, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.05],
	[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
	]).float().cuda().unsqueeze_(0).unsqueeze_(0)
# Training Network #
# Gabor

img_proc = []
for add in training_set:
	print(add)
	x = st.read_image(add,scale_factor,_device=_dvc)
	x = st.apply_filters(x,gabor,use_abs=True)
	x = st.pooling(x, c1_pooling_win, c1_pooling_stride)
	#from channels to minibatch
	x.squeeze_(0)
	x.unsqueeze_(1)
	x = st.intensity_lateral_inhibition(x,inhibition_kernel)
	#from minibatch to channel
	x.squeeze_(1)
	x.unsqueeze_(0)
	x = st.intensity_to_latency(x, timesteps)
	x = st.pointwise_inhibition(x[0])
	st.fire_(x)
	pad_width = math.ceil(math.fabs(min(x.size(-1)-receptive_field,0))/2)
	pad_height = math.ceil(math.fabs(min(x.size(-2)-receptive_field,0))/2)
	img_proc.append(st.pad(x,(pad_width,pad_width,pad_height,pad_height)).cpu()) #cache inputs on ram

img_proc_test = []
for add in testing_set:
	print(add)
	x = st.read_image(add,scale_factor,_device=_dvc)
	x = st.apply_filters(x,gabor,use_abs=True)
	x = st.pooling(x, c1_pooling_win, c1_pooling_stride)
	#from channels to minibatch
	x.squeeze_(0)
	x.unsqueeze_(1)
	x = st.intensity_lateral_inhibition(x,inhibition_kernel)
	#from minibatch to channel
	x.squeeze_(1)
	x.unsqueeze_(0)
	x = st.intensity_to_latency(x, timesteps)
	x = st.pointwise_inhibition(x[0])
	st.fire_(x)
	pad_width = math.ceil(math.fabs(min(x.size(-1)-receptive_field,0))/2)
	pad_height = math.ceil(math.fabs(min(x.size(-2)-receptive_field,0))/2)
	img_proc_test.append(st.pad(x,(pad_width,pad_width,pad_height,pad_height)).cpu()) #cache inputs on ram

# S1 Features #
feature,cstride = st.get_deep_feature(feature, cstride, (5,5), (1,1), gabor)
# C1 Features #
feature,cstride = st.get_deep_feature(feature, cstride, c1_pooling_win, c1_pooling_stride)

mean = torch.ones(number_of_features,4,receptive_field,receptive_field) * 0.8
std = torch.ones(number_of_features,4,receptive_field,receptive_field) * 0.05
weights = torch.normal(mean,std)
if _dvc == 'cuda':
	weights = weights.cuda()

# initial adaptive learning rates
apr_adapt = ((1.0 - 1.0 / len(dataset)) * adaptive_int + adaptive_min) * apr
anr_adapt = ((1.0 - 1.0 / len(dataset)) * adaptive_int + adaptive_min) * anr
app_adapt = ((1.0 / len(dataset)) * adaptive_int + adaptive_min) * app
anp_adapt = ((1.0 / len(dataset)) * adaptive_int + adaptive_min) * anp

best_perf = np.array([0,0,0,0]) # correct, wrong, silence, iteration
best_test = np.array([0,0,0,0]) # correct, wrong, silence, iteration
perf = np.array([0,0,0,0]) # correct, wrong, silence, iteration

to_be_dropped = torch.bernoulli(p_drop).nonzero()
iteration_idx = 0
if need_training:
	ordering = torch.randperm(len(training_set))
	iteration = 800 * len(training_set)
	image_idx = 0
	for i in range(iteration):
		if image_idx == len(training_set):
			iteration_idx += 1
			print("iteration:", iteration_idx)
			#compute performance
			perf = perf/len(training_set)
			perf[-1] = i
			#update adaptive learning rates
			apr_adapt = apr * (perf[1] * adaptive_int + adaptive_min);
			anr_adapt = anr * (perf[1] * adaptive_int + adaptive_min);
			app_adapt = app * (perf[0] * adaptive_int + adaptive_min);
			anp_adapt = anp * (perf[0] * adaptive_int + adaptive_min);

			if best_perf[0] <= perf[0]:
				best_perf = perf
			print("current   :", perf)
			print("best train:", best_perf)
			if perf[2] > 0:
				break
			perf = np.array([0,0,0,0])

			#Testing
			#print("testing...")
			for image_idx in range(len(testing_set)):
				if _dvc == 'cuda':
					img = img_proc_test[image_idx].cuda()
				else:
					img = img_proc_test[image_idx]

				x =	st.convolution(img, weights)
				st.threshold_(x, threhsold)
				spikes = st.fire(x)
				winners = st.get_k_winners(x,kwta,1)
				prepost = st.get_pre_post_ordering(img,spikes,winners[0],tuple(weights[0].shape))
				if len(winners[0]) != 0:
					signal = (neuron_labels[winners[0][0][0]] == testing_lbl[image_idx])
					if signal:
						perf[0]+=1
					else:
						perf[1]+=1
				else:
					perf[2]+=1
			#compute performance
			perf = perf/len(testing_set)
			perf[-1] = i
			if best_test[0] <= perf[0]:
				best_test = perf
				torch.save(weights, "MiladShallow_S2_"+task+".pt")
			print("best test :", best_test)
			print()
			perf = np.array([0,0,0,0])
			to_be_dropped = torch.bernoulli(p_drop).nonzero() #renew dropout for the next iteration
			image_idx = 0
			
		if _dvc == 'cuda':
			img = img_proc[ordering[image_idx]].cuda()
		else:
			img = img_proc[ordering[image_idx]]

		x =	st.convolution(img, weights)
		
		#dropout
		st.feature_inhibition_(x,to_be_dropped)

		st.threshold_(x, threhsold)
		spikes = st.fire(x)
		winners = st.get_k_winners(x,kwta,1)
		prepost = st.get_pre_post_ordering(img,spikes,winners[0],tuple(weights[0].shape))
		if len(winners[0]) != 0:
			signal = (neuron_labels[winners[0][0][0]] == training_lbl[ordering[image_idx]])
			if signal:
				st.STDP(weights, prepost, winners[0], apr_adapt, anr_adapt)
				perf[0]+=1
			else:
				st.STDP(weights, prepost, winners[0], anp_adapt, app_adapt)
				perf[1]+=1
		else:
			perf[2]+=1

		image_idx += 1

	# S1 Features #
	weights = torch.load("MiladShallow_S2_"+task+".pt")
	feature,cstride = st.get_deep_feature(feature, cstride, (receptive_field,receptive_field), (1,1), weights)
	for i in range(number_of_features):
		st.plot_tensor_in_image('output/feature_s2_'+str(i).zfill(4)+'.png',feature[i])

	f = open('perf_milad.txt', 'w')
	f.write(str(best_test))
	f.close()

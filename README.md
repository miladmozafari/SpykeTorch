# SpykeTorch
High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.
![alt text](https://raw.githubusercontent.com/miladmozafari/SpykeTorch/master/logo.png)

SpykeTorch is a Pytorch-based simulator of convolutional spiking neural networks, in which the neurons emit at most one spike per stimulus. SpykeTorch supports STDP and Reward-modulated STDP learning rules. The current code is the early object oriented version of this simulator and you can find the documentation in docs folder. Since SpykeTorch is fully compatible with Pytorch, you can easily use it if you know Pytorch. Soon we will publish a tutorial on how to use SpykeTorch.

MozafariShallow.py is the reimplementation of the paper "First-Spike-Based Visual Categorization Using Reward-Modulated STDP" (https://ieeexplore.ieee.org/document/8356226/), and MozafariDeep.py is the reimplementation of the paper "Combining STDP and Reward-Modulated STDP in Deep Convolutional Spiking Neural Networks for Digit Recognition" (https://arxiv.org/abs/1804.00227v1), using SpykeTorch.


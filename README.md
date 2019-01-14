# SpykeTorch
High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.
![alt text](https://raw.githubusercontent.com/miladmozafari/SpykeTorch/master/logo.png)

SpykeTorch is a Pytorch-based simulator of convolutional spiking neural networks, in which the neurons emit at most one spike per stimulus. SpykeTorch supports STDP and Reward-modulated STDP learning rules. The current code is the early object oriented version of this simulator and unfortunatly is not well-documented. Since SpykeTorch is fully compatible with Pytorch, you can easily use it if you know Pytorch. Our plan is to publish the final code as an easy-to-use framework for simulating the aformentioned type of spiking neural networks.

MozafariShallow.py is the reimplementation of the paper "First-Spike-Based Visual Categorization Using Reward-Modulated STDP" (https://ieeexplore.ieee.org/document/8356226/), using SpykeTorch.

# SpykeTorch
High-speed simulator of convolutional spiking neural networks with at most one spike per neuron.
![alt text](https://raw.githubusercontent.com/miladmozafari/SpykeTorch/master/logo.png)

SpykeTorch is a PyTorch-based simulator of convolutional spiking neural networks, in which the neurons emit at most one spike per stimulus. SpykeTorch supports STDP and Reward-modulated STDP learning rules. The current code is the early object oriented version of this simulator and you can find the documentation in docs folder in PDF format or in our lab website (http://cnrl.ut.ac.ir/SpykeTorch/doc/) in HTML format. Since SpykeTorch is fully compatible with PyTorch, you can easily use it if you know PyTorch. A tutorial is available in the preprint titled "SpykeTorch: Efficient Simulation of Convolutional Spiking Neural Networks with at most one Spike per Neuron" which introduces the SpykeTorch package (https://arxiv.org/abs/1903.02440).

**Scripts info:**
 - [`MozafariShallow.py`](MozafariShallow.py): Reimplementation of the paper "First-Spike-Based Visual Categorization Using Reward-Modulated STDP" (https://ieeexplore.ieee.org/document/8356226/).
 - [`MozafariDeep.py`](MozafariDeep.py): Reimplementation of the paper "Bio-Inspired Digit Recognition Using Reward-Modulated Spike-Timing-Dependent Plasticity in Deep Convolutional Networks" (https://www.sciencedirect.com/science/article/abs/pii/S0031320319301906).
 - [`KheradpishehDeep.py`](KheradpishehDeep.py): Reimplementation of the paper "STDP-based spiking deep convolutional neural networks for object recognition" (https://www.sciencedirect.com/science/article/pii/S0893608017302903).
 - [`tutorial.ipynb`](tutorial.ipynb): A brief tutorial on designing, training, and evaluating a SNN with SpykeTorch.


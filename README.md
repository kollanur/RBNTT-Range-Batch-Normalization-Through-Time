# RBNTT-Range-Batch-Normalization-Through-Time
The following repository is slightly modified version of BNTT ( Batch Normalization Through Time )
[arXiv preprint arXiv:2010.01729](https://arxiv.org/abs/2010.01729)
Instead of using normal Batch Normalization, Range Batchnormalization is used.

## Introduction
 Batch Normalization Through Time (BNTT) technique is used to train the Spiking Neural Network. The proposed approach by [arXiv preprint arXiv:2010.01729](https://arxiv.org/abs/2010.01729) decouples the parameters in a BNTT layer along the time axis to capture the temporal dynamics of spikes.
  The temporally evovling learnable parameters in BNTT allow a neuron to control its spike rate through different time-steps, enabling low-latency and low-energy training from scratch.
  Batch Normalization is hard to implement in FPGA, so instead of using Batch Normalization, we can use Hardware friendly Batch Normalization i.e., Range Batch Normalization.\
  \
  The following repository consists of Range Batch Normalization layer implemented from scratch along with the other techniques mentioned in the [arXiv preprint arXiv:2010.01729](https://arxiv.org/abs/2010.01729).
  
  ## Prerequisites
  * Python 3.6+
  * PyTorch 1.5+
  * NVIDIA GPU (>+ 12GB)
  
  
  ## Training and testing
* VGG9/VGG11 architectures on CIFAR10/CIAR100 datasets

* ```train.py```: code for training  
* ```model.py```: code for VGG9/VGG11 Spiking Neural Networks with BNTT  
* ```utill.py```: code for accuracy calculation / learning rate scheduler

* Run the following command for VGG9 SNN on CIFAR10
```
python train.py --num_steps 25 --lr 0.3 --arch 'vgg9' --dataset 'cifar10' --batch_size 256 --leak_mem 0.95 --num_workers 4 --num_epochs 100
```
* Run the following command for VGG11 SNN on CIFAR100
```
python train.py --num_steps 30 --lr 0.3 --arch 'vgg11' --dataset 'cifar100' --batch_size 128 --leak_mem 0.99 --num_workers 4 --num_epochs 100
```

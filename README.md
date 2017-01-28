# What's this
Implementation of gated convolutional networks by chainer.  
This script solves bits inversion by gated convolutional networks.

# Bits inversion

    000→000
    001→001
    010→011  
    011→010  
    100→111  
    101→110  
    110→100  
    111→101  
  
# Dependencies

    git clone https://github.com/nutszebra/gated_convolutional_networks.git
    cd gated_convolutional_networks
    git submodule init
    git submodule update

# How to run
    python main.py -g 0

# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for some parts.

* Learning rate schedule  
Initial learning rate is 0.1 and learning rate is divided by 10 at [20, 30] epochs. The total number of epochs is 40

* Resblock  
Each resblock contains 2 Gated-Linaer-Unit (GLU)  

* Network  
Ten residual blocks with 16 units and 4 kernel width

* Weight Normalization  
Not implemented  

* Optimization  
Momentum SGD with 0.99 momentum  

* Gradient Clipping  
0.1  

* Weight decay  
0.0001  


# References
Language Modeling with Gated Convolutional Networks [[1]][Paper]  

[paper]: https://arxiv.org/abs/1612.08083 "Paper"

# Neural network
Neural network implemented in numpy. 

## Overview
Project written mainly for university classes. It includes implementation of dense, convolution, pooling, flatten and dropout layer, different optimizers,  weight initializers and train callbacks. For research purpose it allows to conduct some experiments to compare specific parameters of MLP and CONV nets. Project does not support GPU runtime.

## Installation guide

To run project you must have Anaconda installed. 
Clone the repository:
```
git clone https://github.com/Joanna065/neural-network.git
```
Create new conda environment for this project:
```
conda create -n <net-env> python=3.7
conda activate <net-env>
```
Install requirements listed in `requirements.txt` file via conda or pip:
```
while read requirement; do conda install --yes $requirement; done < requirements.txt
```
```
pip install -r requirements.txt
```
Project uses mnist dataset consisting of number images. Download data from github: 
`https://github.com/mnielsen/neural-networks-and-deep-learning.git`. Create file named `user_settings.py` in project root directory and save there absolute path for dataset `DATA_PATH` and directory for results of experiments `SAVE_PATH` to be saved. Example:
```
DATA_PATH = '/home/joanna/lab/neural_net/data/mnist.pkl'
SAVE_PATH = '/home/joanna/lab/neural_net/results/exp_mlp'
```


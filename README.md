# Neural network
Neural network implemented in numpy. 

## Overview
Project written mainly for university classes. It includes implementation of dense and convolution layer, different optimizers,  weight initializers and callbacks during training proccess. For research purpose it allows to conduct some experiments to compare specific parameters of MLP and CONV nets. Project does not support GPU runtime.

## Installation guide

To run project you must have Anaconda installed. 
Clone the repository:
```
git clone https://github.com/Joanna065/neural-network.git
```
Create new conda environment for this project:
```
conda create -n <net-env> python=3.7
```
Install requirements listed in `requirements.txt` file via conda or pip:
```
while read requirement; do conda install --yes $requirement; done < requirements.txt
```
```
pip install -r requirements.txt
```

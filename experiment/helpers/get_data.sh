#!/bin/sh

mkdir data
cd data

# Get mnist
mkdir mnist
wget -O mnist/mnist.data.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget -O mnist/mnist.data.t.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d mnist/mnist.data.bz2
bzip2 -d mnist/mnist.data.t.bz2

# Get cifar 
mkdir cifar
wget -O cifar/cifar.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
cd cifar
tar xzf cifar.tar.gz
rm cifar.tar.gz
mv cifar-10-batches-py/data_batch_* .
mv cifar-10-batches-py/test_batch .
rm -r cifar-10-batches-py

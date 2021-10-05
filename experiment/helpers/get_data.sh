#!/bin/sh

mkdir data
cd data

# Get mnist
mkdir mnist
wget -O mnist/mnist.data.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2
wget -O mnist/mnist.data.t.bz2 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2
bzip2 -d mnist/mnist.data.bz2
bzip2 -d mnist/mnist.data.t.bz2

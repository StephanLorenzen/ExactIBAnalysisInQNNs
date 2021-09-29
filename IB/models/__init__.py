from .models import MNIST,MNIST_FC,CIFAR,CIFAR_FC,ShwartzZiv99
from .activations import get_activation_bound

def load(model_name):
    return {
        'SYN':ShwartzZiv99,
        'MNIST':MNIST,
        'MNIST-FC':MNIST_FC,
        'CIFAR':CIFAR,
        'CIFAR-FC':CIFAR_FC,
    }[model_name]

from .models import MNIST_10,MNIST_BN,ShwartzZiv99
from .activations import get_activation_bound

def load(model_name):
    return {
        'SYN':ShwartzZiv99,
        'MNIST-10':MNIST_10,
        'MNIST-Bottleneck':MNIST_BN,
    }[model_name]

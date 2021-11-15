from .models import MNIST_4X10,MNIST_BN2,MNIST_BN4,MNIST_HG,MNIST_CONV,ShwartzZiv99
from .activations import get_activation_bound

def load(model_name):
    return {
        'SYN':ShwartzZiv99,
        'MNIST-4x10':MNIST_4X10,
        'MNIST-Bottleneck-2':MNIST_BN2,
        'MNIST-Bottleneck-4':MNIST_BN4,
        'MNIST-HourGlass':MNIST_HG,
        'MNIST-Conv':MNIST_CONV,
    }.get(model_name,None)

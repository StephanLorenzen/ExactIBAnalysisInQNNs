from .models import MNIST,ShwartzZiv99
from .activations import get_activation_bound

def load(model_name):
    return {
        'SYN':ShwartzZiv99,
        'MNIST':MNIST,
    }[model_name]

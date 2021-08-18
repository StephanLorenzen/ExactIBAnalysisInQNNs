from .models import ShwartzZiv99
from .activations import get_activation_bound

def load(model_name):
    return {
        'ShwartzZiv99':ShwartzZiv99
    }[model_name]

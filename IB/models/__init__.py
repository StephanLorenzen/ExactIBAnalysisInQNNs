from .models import shwartz_ziv_99
from .activations import get_activation_bound

def load(model_name):
    return {
        'shwartz_ziv_99':shwartz_ziv_99
    }[model_name]

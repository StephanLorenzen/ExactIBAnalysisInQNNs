from .models import shwartz_ziv_99

def load(model_name, activation='tanh'):
    if activation is None:
        raise Exception("Missing or unknown activation function...")
    model = {
        'shwartz_ziv_99':shwartz_ziv_99
    }[model_name]
    return model(activation=activation)

import tensorflow as tf
import tensorflow_model_optimization as tfmo
from tensorflow_model_optimization.quantization.keras.quantizers import LastValueQuantizer, MovingAverageQuantizer

# Fixed range quantizers: use only for bounded (named) activation functions
class DefaultQuantizer(tfmo.quantization.keras.quantizers.Quantizer):
    def __init__(self, num_bits):
        self.num_bits = num_bits
    def build(self, tensor_shape, name, layer):
        return {}
    def get_config(self):
        return {}
## tanh
class TanhQuantizer(DefaultQuantizer):
    def __init__(self, num_bits):
        super().__init__(num_bits=num_bits)
    def __call__(self, inputs, training, weights, **kwargs):
        # Compute activations
        acts = tf.keras.activations.tanh(inputs)
        mn = tf.reduce_min(acts)
        mx = tf.reduce_max(acts)
        return tf.quantization.fake_quant_with_min_max_vars(
                acts, mn, mx, num_bits=self.num_bits, narrow_range=False)

# Default quantization for dense layers
class DefaultDenseQuantizeConfig(tfmo.quantization.keras.QuantizeConfig):
    def __init__(self, num_bits):
        self.num_bits = num_bits
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=self.num_bits, symmetric=False, narrow_range=False, per_axis=False))]
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=self.num_bits, symmetric=False, narrow_range=False, per_axis=False))]
    def set_quantize_weights(self, layer, quantize_weights):
        layer.kernel = quantize_weights[0]
    def set_quantize_activations(self, layer, quantize_activations):
        layer.activation = quantize_activations[0]
    def get_output_quantizers(self, layer):
        return []
    def get_config(self):
        return {}
# Default, but with no weight quantization
class NoWeightDenseQuantizeConfig(DefaultDenseQuantizeConfig):
    def __init__(self, num_bits):
        super().__init__(num_bits=num_bits)
        #self.num_bits = num_bits
    def get_weights_and_quantizers(self, layer):
        return []
    def set_quantize_weights(self, layer, quantize_weights):
        return

# Tanh
class TanhQuantizeConfig(NoWeightDenseQuantizeConfig):
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, TanhQuantizer(num_bits=self.num_bits))]
# Tanh configs
class Tanh4BitConfig(TanhQuantizeConfig):
    def __init__(self):
        super().__init__(num_bits=4)
class Tanh8BitConfig(TanhQuantizeConfig):
    def __init__(self):
        super().__init__(num_bits=8)

# Default configs
class Default4BitConfig(DefaultDenseQuantizeConfig):
    def __init__(self):
        super().__init__(num_bits=4)
class Default8BitConfig(DefaultDenseQuantizeConfig):
    def __init__(self):
        super().__init__(num_bits=8)

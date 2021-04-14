import tensorflow as tf
import tensorflow_model_optimization as tfmo
from tensorflow_model_optimization.quantization.keras.quantizers import LastValueQuantizer, MovingAverageQuantizer

# Fixed range quantizers: use only for bounded (named) activation functions
class DefaultQuantizer(tfmo.quantization.keras.quantizers.Quantizer):
    def build(self, tensor_shape, name, layer):
        return {}
    def get_config(self):
        return {}
## ReLU6
class ReLU6Quantizer(DefaultQuantizer):
    def __call__(self, inputs, training, weights, **kwargs):
        # Compute activations        
        acts = tf.nn.relu6(inputs)
        return tf.quantization.fake_quant_with_min_max_vars(
                acts, 0.0, 6.0, num_bits=8, narrow_range=False)
## tanh
class TanhQuantizer(DefaultQuantizer):
    def __call__(self, inputs, training, weights, **kwargs):
        # Compute activations
        acts = tf.keras.activations.tanh(inputs)
        mn = tf.reduce_min(acts)
        mx = tf.reduce_max(acts)
        return tf.quantization.fake_quant_with_min_max_vars(
                acts, mn, mx, num_bits=8, narrow_range=False)

## SoftMax
class SoftMaxQuantizer(DefaultQuantizer):
    def __call__(self, inputs, training, weights, **kwargs):
        # Compute activations        
        acts = tf.keras.activations.softmax(inputs)
        return tf.quantization.fake_quant_with_min_max_vars(
                acts, 0.0, 1.0, num_bits=8, narrow_range=False)


# Default quantization for dense layers
class DefaultDenseQuantizeConfig(tfmo.quantization.keras.QuantizeConfig):
    def get_weights_and_quantizers(self, layer):
        return [(layer.kernel, LastValueQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, MovingAverageQuantizer(num_bits=8, symmetric=False, narrow_range=False, per_axis=False))]
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
  def get_weights_and_quantizers(self, layer):
    return []
  def set_quantize_weights(self, layer, quantize_weights):
    return

# ReLU6
class ReLU6QuantizeConfig(NoWeightDenseQuantizeConfig):
  def get_activations_and_quantizers(self, layer):
    return [(layer.activation, ReLU6Quantizer())]
# Tanh
class TanhQuantizeConfig(NoWeightDenseQuantizeConfig):
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, TanhQuantizer())]
# SoftMax
class SoftMaxQuantizeConfig(NoWeightDenseQuantizeConfig):
    def get_activations_and_quantizers(self, layer):
        return [(layer.activation, SoftMaxQuantizer())]
    



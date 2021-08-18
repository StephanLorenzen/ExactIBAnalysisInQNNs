import sys

from IB.models import load as load_model
from IB.util.estimator import QuantizedEstimator
from IB.experiment import run_experiment

if len(sys.argv)<3:
    print("Missing arguments, quantize.py <act_function> <bits>.")
    sys.exit(1)

# Activation function
act_fun = sys.argv[1]
bits    = int(sys.argv[2])
repeats = 50 

# Model
_model = load_model("ShwartzZiv99")
Model = lambda: (_model(activation=act_fun, quantize=(bits<=16), num_bits=bits), bits<=16)

# MI estimators
estimators = [QuantizedEstimator(bounds="layer", bits=bits)]

res_path = "out/quantized_"+str(bits)+"bit/"+act_fun+"/"

print("Starting binning experiment.")
run_experiment(Model, estimators, repeats=repeats, out_path=res_path)

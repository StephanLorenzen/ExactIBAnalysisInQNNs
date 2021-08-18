import sys

from IB.models import load as load_model
from IB.util.estimator import BinningEstimator, QuantizedEstimator
from IB.experiment import run_experiment

if len(sys.argv)<2:
    print("Missing arguments, binning.py <act_function>.")
    sys.exit(1)

# Activation function
act_fun = sys.argv[1]
repeats = 50

# Model
_model = load_model("ShwartzZiv99")
Model = lambda: (_model(activation=act_fun, quantize=False), False)

# MI estimators
estimators = []
for n_bins in [30,100,256]:
    estimators.append(BinningEstimator("uniform", n_bins=n_bins, bounds="global"))

res_path = "out/binning/"+act_fun+"/"

print("Starting binning experiment.")
run_experiment(Model, estimators, repeats=repeats, out_path=res_path)

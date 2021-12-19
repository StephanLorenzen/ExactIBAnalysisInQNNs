import sys

from IB.models import load as load_model
from IB.util.estimator import BinningEstimator, QuantizedEstimator
from IB.experiment import run_experiment

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("Missing arguments, quantize.py <experiment> <bits> [prefit=0] [repeats=50].")
        sys.exit(1)

    # Activation function
    exp      = sys.argv[1]
    bits     = int(sys.argv[2])
    prefit   = int(sys.argv[3]) if len(sys.argv)>=4 else 0
    repeats  = int(sys.argv[4]) if len(sys.argv)>=5 else 50

    # Model
    _model = None
    if exp[:3]=="SYN":
        _model  = load_model("SYN")
        act_fun = exp[4:].lower()
        dname   = "SYN"
        epochs  = 8000
    elif exp=="MNIST-Tanh" or exp=="MNIST-ReLU":
        act_fun = exp[6:].lower()
        _model  = load_model("MNIST-10")
        dname   = "MNIST"
        epochs  = 3000
    elif exp[:5]=="MNIST":
        act_fun = "relu"
        _model  = load_model(exp)
        dname   = "MNIST"
        epochs  = 3000
    if _model is None:
        print("Unknown experiment or model!")
        sys.exit(1)
    Model   = lambda: (_model(activation=act_fun, quantize=(bits<=16), num_bits=bits), (bits<=16))

    # MI estimators
    estimators = [QuantizedEstimator(bounds="layer", bits=bits)]

    res_path = "out/quantized/"+exp+("_prefit_"+str(prefit) if prefit>0 else "")+"/"+str(bits)+"/"

    print("Starting quantized experiment.")
    run_experiment(Model, estimators, dname, epochs=epochs, prefit_random=prefit, repeats=repeats, low_memory=(dname!="SYN"), out_path=res_path)

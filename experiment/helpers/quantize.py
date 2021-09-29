import sys

from IB.models import load as load_model
from IB.util.estimator import BinningEstimator, QuantizedEstimator
from IB.experiment import run_experiment

if __name__ == '__main__':
    if len(sys.argv)<3:
        print("Missing arguments, binning.py <experiment> <bits> [use_mean=0] [repeats=50].")
        sys.exit(1)

    # Activation function
    exp      = sys.argv[1]
    bits     = int(sys.argv[2])
    use_mean = (sys.argv[3]=="1") if len(sys.argv)>=4 else False
    repeats  = int(sys.argv[4]) if len(sys.argv)>=5 else 50

    print("Using mean: "+str(use_mean))
    
    if exp not in ("MNIST","MNIST-Tanh","MNIST-ReLU","CIFAR","SYN-Tanh","SYN-ReLU"):
        print("Experiment must be one of 'MNIST[-{Tanh,ReLU}]', 'CIFAR' or 'SYN-{Tanh,ReLU}'")
        sys.exit(1)

    # Model
    if exp[:3]=="SYN":
        _model  = load_model("SYN")
        act_fun = exp[4:].lower()
        dname   = "SYN"
        Model   = lambda: (_model(activation=act_fun, quantize=(bits<=16), num_bits=bits), (bits<=16))
        epochs  = 8000
    elif exp=="MNIST-Tanh" or exp=="MNIST-ReLU":
        act_fun = exp[6:].lower()
        _model  = load_model("MNIST-FC")
        dname   = "MNIST"
        Model   = lambda: (_model(activation=act_fun,quantize=(bits<=16), num_bits=bits), (bits<=16))
        epochs  = 3000
    else:
        _model = load_model(exp)
        dname  = exp
        Model  = lambda: (_model(quantize=(bits<=16), num_bits=bits), (bits<=16))
        epochs = 20

    # MI estimators
    estimators = [QuantizedEstimator(bounds="layer", bits=bits, use_mean=use_mean)]

    res_path = "out/quantized/"+exp+("-mean" if use_mean else "")+"/"+str(bits)+"/"

    print("Starting quantized experiment.")
    run_experiment(Model, estimators, dname, epochs=epochs, repeats=repeats, low_memory=True, out_path=res_path)

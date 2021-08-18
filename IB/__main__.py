import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

if __name__ == '__main__':
    import argparse
    from time import time

    from .models import load as load_model
    from .models import get_activation_bound
    from .util.estimator import BinningEstimator, QuantizedEstimator, KNNEstimator

    from .util import plot as IBplot

    from .experiment import run_experiment

    def experiment(args):
        quantize = args.q
        bits     = args.b
        mi_est   = "quant_"+str(bits) if quantize else args.mi

        # Model
        _model = load_model(args.n)
        Model = lambda: (_model(activation=args.af, quantize=(quantize and bits<=16), num_bits=bits), (quantize and bits<=16))
        (lw,up) = get_activation_bound(args.af)

        if quantize:
            MI_estimators = [QuantizedEstimator(bounds="layer", bits=bits)]
        elif args.mi=="knn":
            MI_estimators = [KNNEstimator(k=10)]
        elif args.mi=="binning":
            MI_estimators = []
            for n_bins in [30, 100, 256]:
                MI_estimators.append(BinningEstimator("uniform", n_bins=n_bins, bounds="layer"))

        out_name = str(int(time())) if args.en is None else args.en
        out_path = args.o+("" if args.o[-1:]=="/" else "/")+out_name+"/"

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        print("Running experiment!")
        print("> model = '"+args.n+"', activation = '"+args.af+"'")
        print("> epochs = "+str(args.e)+", learning rate = "+str(args.lr))
        print("> quantization = "+("Yes" if args.q else "No"))
        print("> MI estimator ='"+mi_est+"'")
        print("> repeats = "+str(args.r))
        # Write info to file as well 
        with open(out_path+"info.txt", "w") as info:
            info.write("model:        "+str(args.n)+"\n")
            info.write("activation:   "+str(args.af)+"\n")
            info.write("epochs:       "+str(args.e)+"\n")
            info.write("lr:           "+str(args.lr)+"\n")
            info.write("quantization: "+("Yes" if args.q else "No")+"\n")
            info.write("bits:         "+str(args.b)+"\n")
            info.write("data:         "+str(args.d)+"\n")
            info.write("estimator:    "+str(mi_est)+"\n")
            info.write("repeats:      "+str(args.r)+"\n")

        first_rep = args.sf
        if first_rep > 1:
            print("> Restarting from repetition "+str(first_rep))
        run_experiment(
                Model,
                MI_estimators,
                data         = args.d,
                lr           = args.lr, 
                batch_size   = 256,
                epochs       = args.e, 
                repeats      = args.r,
                out_path     = out_path,
                start_from   = first_rep,
                use_carbontracker = False 
                )

    def plot(args):
        name = args.name
        in_path = args.o+("/" if args.o[-1:]!="/" else "")+name+"/"
        if not os.path.isdir(in_path):
            raise Exception("Unknown path or experiment: '"+in_path+"'")
        if args.type=="IB_plane":
            IBplot.information_plane(path=in_path, est=args.mi)
        elif args.type=="MI_X":
            IBplot.mi("X", path=in_path, est=args.mi)
        elif args.type=="MI_Y":
            IBplot.mi("Y", path=in_path, est=args.mi)
        elif args.type=="accuracy":
            IBplot.accuracy(path=in_path)
        elif args.type=="activations":
            IBplot.activations(path=in_path)

    
    parser = argparse.ArgumentParser(
        description="Run experiments for the Information Bottleneck in Deep Learning."
    )
    subparsers = parser.add_subparsers(metavar="COMMAND", help="Choose a command")

    ##### EXPERIMENT
    parser_exp  = subparsers.add_parser("experiment", help="Run experiment")
    # Network setup
    parser_exp.add_argument("-n", metavar="NETWORK", type=str, default="ShwartzZiv99",
                            choices={"ShwartzZiv99"}, help="Network to use.")
    parser_exp.add_argument("-af", metavar="ACT_FUNC", type=str, default="tanh",
                            choices={"tanh","relu","relu6"}, help="Activation function.")
    parser_exp.add_argument("-lr", type=float, default=10**-4,
                            help="Learning rate used in training.")
    parser_exp.add_argument("-e", metavar="EPOCHS", type=int, default=8000, help="Number of epochs.")
    parser_exp.add_argument("-q", action='store_const', const=True, default=False,
                            help="Quantize the model (changes default binning strategy!).")
    parser_exp.add_argument("-b", metavar="BITS", type=int, default=8, help="Number of bits for quantization, if -q set, must be in (4,8).")
    
    # Binning setup
    parser_exp.add_argument("-mi", metavar="ESTIMATOR", type=str,
                            default="binning", choices={"binning", "knn"},
                            help="MI estimator.")

    # Experiment setup
    parser_exp.add_argument("-d", metavar="DATA", type=str, default="tishby",
                            help="Data for experiment")
    parser_exp.add_argument("-r", metavar="REPEATS", type=int, default=10,
                            help="Number of experiment repeats")
    parser_exp.add_argument("-en", metavar="NAME", type=str, default=None,
                            help="Name of experiment, default is auto-generated from parameters.")
    parser_exp.add_argument("-o", metavar="PATH", type=str, default="out/",
                            help="Path to store outputs, default is 'out/'")
    parser_exp.add_argument("-sf", metavar="FIRST_REP", type=int, default=1, help="Starting repatition (default 1).")
    parser_exp.set_defaults(func=experiment)

    ##### PLOT
    parser_plot = subparsers.add_parser("plot", help="Plot IB plane and other statistics")
    parser_plot.add_argument("type", metavar="TYPE", type=str,
                            help="Type of plot", choices={"IB_plane","MI_X","MI_Y","accuracy","activations"})
    parser_plot.add_argument("name", metavar="NAME", type=str, help="Name of experiment")
    parser_plot.add_argument("-mi", metavar="ESTIMATOR", type=str, default="binning_uniform_30",
                            help="Name of estimator to plot for.")
    parser_plot.add_argument("-o", metavar="PATH", type=str, default="out/",
                            help="Path to store outputs, default is 'out/'")
    parser_plot.set_defaults(func=plot)

    args = parser.parse_args()

    args.func(args)

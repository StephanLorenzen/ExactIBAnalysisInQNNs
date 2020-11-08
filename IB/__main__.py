import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import sys

if __name__ == '__main__':
    import argparse
    from time import time

    from .models import load as load_model
    from .models import get_activation_bound
    from .tools.information_theory import estimator as est

    from .tools import plot as IBplot

    from .experiment import run_experiment
    from .analyse import plot_IB_plane

    def experiment(args):
        # Model
        _model = load_model(args.n)
        model = lambda: _model(activation=args.af)
        (lw,up) = get_activation_bound(args.af)

        # MI_estimator
        estimator = { 
                "binning_uniform":   est.binning_uniform,
                "binning_quantized": est.binning_quantized,
                "binning_adaptive":  est.binning_adaptive,
                "knn":               est.knn,
        }.get(args.mi, None)
        est_args = {"n_bins":args.nb,"upper":up,"lower":lw}

        out_name = str(int(time())) if args.en is None else args.en
        out_path = args.o+("" if args.o[-1:]=="/" else "/")+out_name+"/"

        if not os.path.isdir(out_path):
            os.makedirs(out_path)

        print("Running experiment!")
        print("> model = '"+args.n+"', activation = '"+args.af+"'")
        print("> epochs = "+str(args.e)+", learning rate = "+str(args.lr))
        print("> MI estimator ='"+args.mi+"', number of bins = "+str(args.nb))
        print("> repeats = "+str(args.r))
        # Write info to file as well 
        with open(out_path+"info.txt", "w") as info:
            info.write("model:      "+str(args.n)+"\n")
            info.write("activation: "+str(args.af)+"\n")
            info.write("epochs:     "+str(args.e)+"\n")
            info.write("lr:         "+str(args.lr)+"\n")
            info.write("data:       "+str(args.d)+"\n")
            info.write("estimator:  "+str(args.mi)+"\n")
            info.write("n_bins:     "+str(args.nb)+"\n")
            info.write("repeats:    "+str(args.r)+"\n")

        run_experiment(
                model        = model, 
                lr           = args.lr, 
                epochs       = args.e, 
                data         = args.d,
                MI_estimator = (estimator, est_args),
                repeats      = args.r,
                out_path     = out_path
                )

    def plot(args):
        name = args.name
        in_path = args.o+("/" if args.o[-1:]!="/" else "")+name+"/"
        if not os.path.isdir(in_path):
            raise Exception("Unknown path or experiment: '"+in_path+"'")
        if args.type=="IB_plane":
            IBplot.information_plane(path=in_path)
        elif args.type=="MI_X":
            IBplot.mi("X", path=in_path)
        elif args.type=="MI_Y":
            IBplot.mi("Y", path=in_path)

    #def convert(args):
    #    pass


    parser = argparse.ArgumentParser(
        description="Run experiments for the Information Bottleneck in Deep Learning."
    )
    subparsers = parser.add_subparsers(metavar="COMMAND", help="Choose a command")

    ##### EXPERIMENT
    parser_exp  = subparsers.add_parser("experiment", help="Run experiment")
    # Network setup
    parser_exp.add_argument("-n", metavar="NETWORK", type=str, default="shwartz_ziv_99",
                            choices={"shwartz_ziv_99"}, help="Network to use.")
    parser_exp.add_argument("-af", metavar="ACT_FUNC", type=str, default="tanh",
                            choices={"tanh","ReLU"}, help="Activation function.")
    parser_exp.add_argument("-lr", type=float, default=10**-4,
                            help="Learning rate used in training.")
    parser_exp.add_argument("-e", metavar="EPOCHS", type=int, default=8000, help="Number of epochs.")
    
    # Binning setup
    parser_exp.add_argument("-mi", metavar="ESTIMATOR", type=str,
                            default="binning_uniform", choices={"binning_uniform"},
                            help="MI estimator.")
    parser_exp.add_argument("-nb", metavar="BINS", type=int, default=30, help="Number of bins.")

    # Experiment setup
    parser_exp.add_argument("-d", metavar="DATA", type=str, default="tishby",
                            help="Data for experiment")
    parser_exp.add_argument("-r", metavar="REPEATS", type=int, default=10,
                            help="Number of experiment repeats")
    parser_exp.add_argument("-en", metavar="NAME", type=str, default=None,
                            help="Name of experiment, default is auto-generated from parameters.")
    parser_exp.add_argument("-o", metavar="PATH", type=str, default="out/",
                            help="Path to store outputs, default is 'out/'")
    parser_exp.set_defaults(func=experiment)

    ##### PLOT
    parser_plot = subparsers.add_parser("plot", help="Plot IB plane and other statistics")
    parser_plot.add_argument("type", metavar="TYPE", type=str,
                            help="Type of plot", choices={"IB_plane","MI_X","MI_Y","accuracy"})
    parser_plot.add_argument("name", metavar="NAME", type=str, help="Name of experiment")
    parser_plot.add_argument("-o", metavar="PATH", type=str, default="out/",
                            help="Path to store outputs, default is 'out/'")
    parser_plot.set_defaults(func=plot)

    ##### CONVERT
    #parser_conv = subparsers.add_parser("convert", help="Convert output to various formats, e.g. pdflatex input")
    #parser_conv.set_defaults(func=convert)

    args = parser.parse_args()

    args.func(args)

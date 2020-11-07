import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

if __name__ == '__main__':
    import argparse

    from .experiment import run
    from .analyse import plot_IB_plane

    def experiment(args):
        print("Main experiment, "+str(args))
        pass

    def plot(args):
        pass

    def convert(args):
        pass


    parser = argparse.ArgumentParser(description="Run experiments for the Information Bottleneck in Deep Learning.")
    subparsers = parser.add_subparsers(metavar="COMMAND", help="Choose a command")

    ##### EXPERIMENT
    parser_exp  = subparsers.add_parser("experiment", help="Run experiment")
    # Network setup
    parser_exp.add_argument("-n", metavar="NETWORK", type=str, default="shwartz_ziv_99",
                            choices={"shwartz_ziv_99"}, help="Network to use.")
    parser_exp.add_argument("-lr", type=float, default=10**-4,
                            help="Learning rate used in training.")
    parser_exp.add_argument("-e", metavar="EPOCHS", type=int, default=8000, help="Number of epochs.")
    
    # Binning setup
    parser_exp.add_argument("-b", metavar="BINNING", type=str,
                            default="binning_uniform", choices={"binning_uniform"},
                            help="Binning strategy.")
    parser_exp.add_argument("-nb", metavar="BINS", type=int, default=30, help="Number of bins.")

    # Experiment setup
    parser_exp.add_argument("-r", metavar="REPEATS", type=int, default=10, help="Number of experiment repeats")
    parser_exp.add_argument("-en", metavar="NAME", type=str, default=None,
                            help="Name of experiment, default is auto-generated from parameters.")
    parser_exp.add_argument("-o", metavar="PATH", type=str, default="out/",
                            help="Path to store outputs, default is 'out/'")
    parser_exp.set_defaults(func=experiment)

    ##### PLOT
    parser_plot = subparsers.add_parser("plot", help="Plot IB plane and other statistics")
    parser_plot.add_argument("type", metavar="TYPE", type=str, help="Type of plot", choices={"plane","test_err"})
    parser_plot.set_defaults(func=plot)

    ##### CONVERT
    parser_conv = subparsers.add_parser("convert", help="Convert output to various formats, e.g. pdflatex input")
    parser_conv.set_defaults(func=convert)

    args = parser.parse_args()
    args.func(args)

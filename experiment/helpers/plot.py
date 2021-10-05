import sys
import os
import pandas as pd

import IB.util.io as iio

if len(sys.argv)<2:
    print("Missing argument.")
    sys.exit(1)

EXP = sys.argv[1]

OUT_DIR = "helpers/latex/"+EXP+"/"

def latex_MI(est,out_suf,data_path):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+est+".csv"
    print("Creating MI plane file: '"+out_file+"'")
    mean, std = iio.load_MI(data_path,est)
    df  = {"x":[],"y":[],"c":[]}
    if len(mean) == 5000:
        mean = mean[:3000]
        std = std[:3000]
    # 8000 / 5 * 6 = 9600
    # 3000 / 2 * 5 = 7500
    mod = 5 if len(mean)>6000 else 2
    for epoch,mi in enumerate(mean):
        if epoch%mod != 0:
            continue
        df["c"] += [epoch/len(mean)]*len(mi)
        df["x"] += [x for x,_ in mi]
        df["y"] += [y for _,y in mi]
    
    pd.DataFrame(df).to_csv(out_file,index_label="epoch")

def latex_accuracy(out_suf,data_path):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+"accuracy.csv"
    print("Creating '"+out_file+"'")
    df = dict()
    train, test = iio.load_accuracy(data_path)
    df["train_mean"] = train[0]
    df["train_std"] = train[1]
    df["train_up"] = train[0]+train[2]
    df["train_lw"] = train[0]-train[2]
    df["test_mean"] = test[0]
    df["test_std"] = test[1]
    df["test_up"] = test[0]+test[2]
    df["test_lw"] = test[0]-test[2]
    df = pd.DataFrame(df)
    reduce = len(df)//200
    df = df.iloc[::reduce]
    df.to_csv(out_file,index_label="epoch")

if EXP=="binning":
    for nb in [30,100,256]:
        latex_MI("binning_uniform_"+str(nb)+"_global", "SYN-Tanh/", "out/binning/SYN-Tanh/")
        latex_MI("binning_uniform_"+str(nb)+"_global", "SYN-ReLU/", "out/binning/SYN-ReLU/")

elif EXP=="quantize":
    b = "8"
    latex_MI("quantized_layer_"+b+"_bits", "SYN-Tanh/"+b+"/", "out/quantized/SYN-Tanh/"+b+"/")
    latex_MI("quantized_layer_"+b+"_bits", "SYN-ReLU/"+b+"/", "out/quantized/SYN-ReLU/"+b+"/")
    latex_MI("quantized_layer_8_bits", "MNIST-BN/", "out/quantized/MNIST-Bottleneck/8/")

elif EXP=="accuracy":
    latex_accuracy("SYN-Tanh/bin/", "out/binning/SYN-Tanh/")
    latex_accuracy("SYN-ReLU/bin/", "out/binning/SYN-ReLU/")
    for b in ["4","8","32"]:
        latex_accuracy("SYN-Tanh/"+b+"/", "out/quantized/SYN-Tanh/"+b+"/")
        latex_accuracy("SYN-ReLU/"+b+"/", "out/quantized/SYN-ReLU/"+b+"/")
    latex_accuracy("MNIST-BN/", "out/quantized/MNIST-Bottleneck/8/")
    latex_accuracy("MNIST-10/", "out/quantized/MNIST-10/8/")

elif EXP=="quantize-extra":
    # Plots for 4,32-bit quantized synthetic experiments: Tanh and ReLU, I-plane, accuracy
    for b in ["4","32"]:
        latex_MI("quantized_layer_"+b+"_bits", "SYN-Tanh/"+b+"/", "out/quantized/SYN-Tanh/"+b+"/")
        latex_MI("quantized_layer_"+b+"_bits", "SYN-ReLU/"+b+"/", "out/quantized/SYN-ReLU/"+b+"/")
    latex_MI("quantized_layer_8_bits", "MNIST-10/", "out/quantized/MNIST-10/8/")


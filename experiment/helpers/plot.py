import sys
import os
import pandas as pd
import numpy as np

import IB.util.io as iio

if len(sys.argv)<2:
    print("Missing argument.")
    sys.exit(1)

EXP = sys.argv[1]

OUT_DIR = "helpers/latex/"+EXP+"/"

def latex_MI(est,out_suf,data_path,prefit=False):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+est+("_prefit" if prefit else "")+".csv"
    print("Creating MI plane file: '"+out_file+"'")
    try: mean, std = iio.load_MI(data_path,est,load_prefit=prefit)
    except Exception as e:
        print("File not found",(data_path,est,e))
        return
    df  = {"x":[],"y":[],"c":[]}
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

# Return best, center, second worst and worst
def latex_MI_ranks(est,out_suf,data_path):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    try: mean, _ = iio.load_MI(data_path,est,load_prefit=False)
    except:
        print("File not found",(data_path,est))
        return
    repeats = iio.load_MI_repeats(data_path,est,load_prefit=False)
    n_rep, n_epoch, n_layer = repeats.shape
    mean = np.array(mean).reshape(-1,n_layer)
    
    repeats = [(rep,((rep-mean)**2).sum()) for rep in repeats]
    repeats.sort(key=lambda x: x[1])
    repeats = [rep.reshape(-1,n_layer//2,2) for rep,_ in repeats]
    relevant = [
        ("best",repeats[0]),
        ("middle",repeats[n_rep//2]),
        ("2nd_worst", repeats[-2]),
        ("worst", repeats[-1])
    ]
    for name, rep in relevant:
        df  = {"x":[],"y":[],"c":[]}
        # 8000 / 5 = 1600
        # 3000 / 2 = 1500
        mod = 5 if len(mean)>6000 else 2
        for epoch,mi in enumerate(rep):
            if epoch % mod != 0:
                continue
            df["c"] += [epoch/n_epoch]*len(mi)
            df["x"] += [x for x,_ in mi]
            df["y"] += [y for _,y in mi]
        out_file = out_dir+est+"_"+name+".csv"
        print("Creating worst MI plane file: '"+out_file+"'")
        pd.DataFrame(df).to_csv(out_file,index_label="epoch")

def latex_MI_var(est,out_suf,data_path,var="X",prefit=False):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+est+"_"+var+("_prefit" if prefit else "")+".csv"
    print("Creating MI plane file for "+var+": '"+out_file+"'")
    try: mean, std = iio.load_MI(data_path,est,load_prefit=prefit)
    except:
        print("File not found",(data_path,est))
        return
    nl = len(mean[0])
    df = {"epoch":[]}
    for i in range(nl):
        df["mean_"+str(i+1)] = []
        df["up_"+str(i+1)] = []
        df["lw_"+str(i+1)] = []
    # 8000 / 5 = 1600
    # 3000 / 2 = 1500
    mod = 5 if len(mean)>6000 else 2
    didx = 0 if var=="X" else 1
    for epoch,mi_mean in enumerate(mean):
        if epoch % mod != 0:
            continue
        mi_std = std[epoch]
        df["epoch"].append(epoch)
        for i in range(nl):
            df["mean_"+str(i+1)].append(mi_mean[i][didx])
            df["up_"+str(i+1)].append(mi_mean[i][didx]+mi_std[i][didx])
            df["lw_"+str(i+1)].append(mi_mean[i][didx]-mi_std[i][didx])
    pd.DataFrame(df).to_csv(out_file)

def latex_accuracy(out_suf,data_path):
    out_dir = OUT_DIR+out_suf
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+"accuracy.csv"
    print("Creating '"+out_file+"'")
    df = dict()
    try: train, test = iio.load_accuracy(data_path)
    except:
        print("File not found",(data_path))
        return
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

if EXP=="quantize":
    latex_MI("quantized_layer_8_bits", "SYN-Tanh/8/", "out/quantized/SYN-Tanh/8/")
    latex_MI_ranks("quantized_layer_8_bits", "SYN-Tanh/8/", "out/quantized/SYN-Tanh/8/")
    latex_MI("quantized_layer_8_bits", "SYN-ReLU/8/", "out/quantized/SYN-ReLU/8/")
    latex_MI_ranks("quantized_layer_8_bits", "SYN-ReLU/8/", "out/quantized/SYN-ReLU/8/")
    latex_MI("quantized_layer_8_bits", "MNIST-Bottleneck-2/", "out/quantized/MNIST-Bottleneck-2/8/")
    latex_MI_ranks("quantized_layer_8_bits", "MNIST-Bottleneck-2/", "out/quantized/MNIST-Bottleneck-2/8/")
    
    for var in ("X","Y"):
        latex_MI_var("quantized_layer_8_bits", "SYN-Tanh/8/","out/quantized/SYN-Tanh/8/",var)
        latex_MI_var("quantized_layer_8_bits", "SYN-ReLU/8/","out/quantized/SYN-ReLU/8/",var)
        latex_MI_var("quantized_layer_8_bits", "MNIST-Bottleneck-2/","out/quantized/MNIST-Bottleneck-2/8/",var)

elif EXP=="accuracy":
    latex_accuracy("SYN-Tanh/bin/", "out/binning/SYN-Tanh/")
    latex_accuracy("SYN-ReLU/bin/", "out/binning/SYN-ReLU/")
    for b in ["4","8","32"]:
        latex_accuracy("SYN-Tanh/"+b+"/", "out/quantized/SYN-Tanh/"+b+"/")
        latex_accuracy("SYN-ReLU/"+b+"/", "out/quantized/SYN-ReLU/"+b+"/")
    latex_accuracy("MNIST-Bottleneck-2/", "out/quantized/MNIST-Bottleneck-2/8/")
    latex_accuracy("MNIST-Bottleneck-4/", "out/quantized/MNIST-Bottleneck-4/8/")
    latex_accuracy("MNIST-HourGlass/",    "out/quantized/MNIST-HourGlass/8/")
    latex_accuracy("MNIST-4x10/",         "out/quantized/MNIST-4x10/8/")
    latex_accuracy("MNIST-Conv/",         "out/quantized/MNIST-Conv/8/")

elif EXP=="binning":
    for nb in [30,100,256]:
        latex_MI("binning_uniform_"+str(nb)+"_global", "SYN-Tanh/", "out/binning/SYN-Tanh/")
        latex_MI("binning_uniform_"+str(nb)+"_global", "SYN-ReLU/", "out/binning/SYN-ReLU/")

elif EXP=="bit-width":
    # Plots for 4,32-bit quantized synthetic experiments: Tanh and ReLU, I-plane, accuracy
    for b in ["4","32"]:
        latex_MI("quantized_layer_"+b+"_bits", "SYN-Tanh/"+b+"/", "out/quantized/SYN-Tanh/"+b+"/")
        latex_MI("quantized_layer_"+b+"_bits", "SYN-ReLU/"+b+"/", "out/quantized/SYN-ReLU/"+b+"/")

elif EXP=="prefit":
    # Plots for 8bit ReLU and Tanh prefits
    latex_MI("quantized_layer_8_bits", "SYN-Tanh/", "out/quantized/SYN-Tanh_prefit_1000/8/")
    latex_MI("quantized_layer_8_bits", "SYN-Tanh/", "out/quantized/SYN-Tanh_prefit_1000/8/", prefit=True)
    latex_MI("quantized_layer_8_bits", "SYN-ReLU/", "out/quantized/SYN-ReLU_prefit_1000/8/")
    latex_MI("quantized_layer_8_bits", "SYN-ReLU/", "out/quantized/SYN-ReLU_prefit_1000/8/", prefit=True)
    
elif EXP=="archs":
    latex_MI("quantized_layer_8_bits", "MNIST-Bottleneck-4/", "out/quantized/MNIST-Bottleneck-4/8/")
    latex_MI("quantized_layer_8_bits", "MNIST-HourGlass/",    "out/quantized/MNIST-HourGlass/8/")
    latex_MI("quantized_layer_8_bits", "MNIST-4x10/",         "out/quantized/MNIST-4x10/8/")
    latex_MI("quantized_layer_8_bits", "MNIST-Conv/",         "out/quantized/MNIST-Conv/8/")

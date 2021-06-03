import sys
import os
import pandas as pd

import IB.util.io as iio

if len(sys.argv)<2:
    print("Missing argument.")
    sys.exit(1)

EXP = sys.argv[1]

OUT_DIR = "helpers/latex/"+EXP+"/"

def latex_MI(act,est,data_path):
    out_dir = OUT_DIR+act+"/"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+est+".csv"
    print("Creating MI plane file: '"+out_file+"'")
    mean, std = iio.load_MI(data_path+act+"/",est)
    df  = {"x":[],"y":[],"c":[]}
    df_layer = dict([("X"+str(i),[]) for i in range(1,6+1)]+[("Y"+str(i),[]) for i in range(1,6+1)])
    # 8000 /  5 * 6 = 9600
    mod = 5
    for epoch,mi in enumerate(mean):
        if epoch%mod != 0:
            continue
        df["c"] += [epoch/len(mean)]*len(mi)
        df["x"] += [x for x,_ in mi]
        df["y"] += [y for _,y in mi]

    pd.DataFrame(df).to_csv(out_file,index_label="epoch")

def latex_accuracy(acts,data_path,suffix=''):
    out_dir = OUT_DIR
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    out_file = out_dir+"accuracy"+suffix+".csv"
    print("Creating '"+out_file+"'")
    df = dict()
    for act in acts:
        train, test = iio.load_accuracy(data_path+act+"/")
        df[act+"_train_mean"] = train[0]
        df[act+"_train_std"] = train[1]
        df[act+"_train_up"] = train[0]+train[2]
        df[act+"_train_lw"] = train[0]-train[2]
        df[act+"_test_mean"] = test[0]
        df[act+"_test_std"] = test[1]
        df[act+"_test_up"] = test[0]+test[2]
        df[act+"_test_lw"] = test[0]-test[2]
    df = pd.DataFrame(df)
    df = df.iloc[::40]
    df.to_csv(out_file,index_label="epoch")



if EXP=="binning":
    for nb in [30,100,256]:
        latex_MI("relu","binning_uniform_"+str(nb)+"_global", "out/binning/")
        latex_MI("tanh","binning_uniform_"+str(nb)+"_global", "out/binning/")

elif EXP=="quantize":
    latex_MI("tanh", "quantized_layer_8_bits", "out/quantized_8bit/")
    latex_MI("relu", "quantized_layer_8_bits", "out/quantized_8bit/")

elif EXP=="accuracy":
    latex_accuracy(["relu","tanh"], "out/binning/", suffix="_binning")
    latex_accuracy(["relu","tanh"], "out/quantized_8bit/", suffix="_8bit")

elif EXP=="quantize-extra":
    latex_MI("tanh", "quantized_layer_4_bits", "out/quantized_4bit/")
    latex_MI("relu", "quantized_layer_4_bits", "out/quantized_4bit/")
    latex_accuracy(["relu","tanh"],"out/quantized_4bit/", suffix="_4bit")
    latex_MI("tanh", "quantized_layer_32_bits", "out/quantized_32bit/")
    latex_MI("relu", "quantized_layer_32_bits", "out/quantized_32bit/")
    latex_accuracy(["relu","tanh"],"out/quantized_32bit/", suffix="_32bit")

# Prepare Tanh quantized
#latex_MI("tanh_fixed_quantized", "quantized_fixed", "quantized_tanh/")
#latex_MI("tanh_quantized", "quantized_neuron", "quantized_tanh/")


# Accuracy
#latex_accuracy(["relu","tanh"],"accuracy/")

#latex_accuracy(["relu","tanh","relu_quantized","tanh_quantized"],"accuracy/")

# ReLU comparison uniform vs adaptive binning
#latex_MI("relu","binning_uniform_30","relu_binning_comparison/")
#latex_MI("relu","binning_adaptive_30","relu_binning_comparison/")
# ReLU quantized
#latex_MI("relu_quantized","binning_quantized_30","relu_quantized/")

# tanh comparison uniform vs adaptive binning
#latex_MI("tanh","binning_uniform_30","tanh_binning_comparison/")
#latex_MI("tanh","binning_adaptive_30","tanh_binning_comparison/")
# tanh quantized
#latex_MI("tanh_quantized","binning_quantized_30","tanh_quantized/")

#for quant in ["4","32"]:
#    for act in ["relu","tanh"]:
#        latex_MI(act+"_"+quant+"bit", "quantized_layer_"+quant+"_bits", quant+"bit/"+act+"_")
#
#    latex_accuracy(["relu_"+quant+"bit","tanh_"+quant+"bit"], quant+"bit/")


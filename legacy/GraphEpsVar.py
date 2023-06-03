import csv
import pandas
import matplotlib.pyplot as plt
import numpy as np
import random
from decimal import *

from ast import literal_eval

fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax2 = ax1.twiny()

# # RR
colors = ["blue","red", "green", "orange", "purple", "magenta"]
i = 1
file_list = ['2DresultsRREarlySplitVariable.csv'] #, '1DresultsRRSplitTop.csv']#, str(i)+'DresultsRRSplitTop.csv'] #, str(i)+'DresultsRRSplitBottom.csv']

df_raw = pandas.read_csv(file_list[0])
df_raw = df_raw[(df_raw["DB size"] == 100000)]
df_constant = df_raw[(df_raw["Eps var"] == "constant")]
df_linear = df_raw[(df_raw["Eps var"] == "linear")]
df_exp = df_raw[(df_raw["Eps var"] == "exp")]
df_half_exp = df_raw[(df_raw["Eps var"] == "half exp")]

variable_types = {"constant": df_constant, "linear": df_linear, "exp": df_exp, "half exp": df_half_exp}

for key in variable_types.keys():
    df_NN = variable_types[key]
    split_list = [0,2, 4, 8]
    i = 1

    for split in split_list:
        df_NN_count = df_NN[(df_NN["Num Max Splits"] == split)]

        nearest_acc_mill_tree = []
        nearest_acc_mill_ldp = []
        for index, row in df_NN_count.iterrows():
            tree_acc = literal_eval(row['Tree results'])
            nearest_acc_mill_tree.append(tree_acc[0])
        
        label_custom = key + ", " + str(split) + " levels, "
        print(label_custom, df_NN_count["Avg 5 Tree"].iloc[0], df_NN_count["Tree Eps"].iloc[0])
        graph_line = "solid"
       
        ax1.plot(df_NN_count["Tree Eps"], df_NN_count["Avg 5 Tree"], label=label_custom + '(TT)', linestyle=graph_line, color=colors[i-1], marker=".")
        ax1.plot(df_NN_count["Tree Eps"], nearest_acc_mill_tree, label=label_custom + '(TT)', linestyle=graph_line, color=colors[i-1], marker=".")

        # ax2.semilogx(df_NN_count_ldp["Eps/Sensitivity"], df_NN_count_ldp["Avg 5 LDP"], label=label_custom + '(GEO)', linestyle="dotted", color=colors[i-1], marker=".")
        # ax2.semilogx(df_NN_count_ldp["Eps/Sensitivity"], nearest_acc_mill_ldp, label=label_custom + '(GEO)', linestyle="dotted", color=colors[i-1], marker=".")
        i +=1

plt.ylabel("Accuracy (%)")
ax1.set_xlabel("Final epsilon")
# ax2.set_xlabel("Eps/Sensitivity (GEO)")
lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc=4)
ax1.legend(lines, labels)

plt.savefig('2DRREarlyGeo.png')
plt.show()

##This part contunies after VG_nsclc_paper.py,
#It is currently split upped because, running VG_nsclc_paper.py takes a while

#Coppied librareis from previous instance
#Libraries
import os

import Utils as utils
from VG_Functions import *
import numpy as np
import pickle
from os import *
from sklearn.metrics import r2_score
import FitFunctions as ff
import warnings
from scipy.optimize import curve_fit
import sys
import json
#C, this package is needed
import builtins

#Some extra libraries needed for plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import time as TIME

#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)


##initial condtions, same as VG_nsclc_paper.py,
studies = ['a', 'a', 'c', 'd', 'e']
studies =['1', '2', '3', '4', '5']

trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
#C, this got in Narmian code overwritten almost immediatly, dont know exactly why.
functions =['Exponential']

result = pd.DataFrame()


#C, the originial for loop, with slight modifications, however, there are empty "rSquare', leading to code breaking.
#C, So, I comment the orgnial for documentaion sake.
# for f in functions:
#     temp = []
#     indices = []
#     for s in studies:
#         #result_dict = pickle.load( open( r"D:\Spider Project\Fit\080221\\" + f + '\\' + s + ".pkl", "rb" ) )
#         #result_dict = pickle.load( builtins.open( dataset_path +'\\' +  f + '/' + s + ".pkl", "rb" ) ) #C, open always need builtins.open
#         result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f, s + ".pkl"), "rb")) #C, Made it system friendly
#
#         arms = list(result_dict.keys())
#         arms.sort()
#         for arm in arms:
#             for trend in trends:
#                 indices.append(arm + '_' + trend)
#
#                 #C, debugging
#                 # Existing code
#                 rSquare_values = result_dict[arm][trend]['rSquare']
#                 # New print statement for debugging
#                 print("rSquare values:", rSquare_values)
#
#                 temp.append(np.around(np.nanmean(result_dict[arm][trend]['rSquare']), 3))
#     result[f] = temp

####################################################################################################################







#C, edited for loop so that the empty rSquared get removed from the normal one, but from the copy it get replaced with zeros
result = pd.DataFrame()
result_with_zeros = pd.DataFrame()
for f in functions:
    temp = []
    temp_with_zeros = []
    indices = []
    indices_with_zeros = []
    for s in studies:
        # Use os.path.join to make the code system-independent
        print(f'path::{os.path.join(dataset_path, f,"Seed_24", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f,"Seed_24", s + ".pkl"), "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                index_name = arm + '_' + trend
                indices_with_zeros.append(index_name)

                print(f"Index_name:{index_name}")

                rSquare_values = result_dict[arm][trend]['rSquare']
                #print(f"result_dict:{result_dict[arm]}")
                #TIME.sleep(10)

                # Check if rSquare_values is empty
                if not rSquare_values:
                    print("rSquare values is empty for arm:", arm, ", trend:", trend)
                    temp_with_zeros.append(0.0)
                    continue

                print("rSquare values:", rSquare_values)
                temp.append(np.around(np.nanmean(rSquare_values), 3))
                temp_with_zeros.append(np.around(np.nanmean(rSquare_values), 3))
                indices.append(index_name)

    result[f] = temp
    result_with_zeros[f] = temp_with_zeros


#
# result = pd.DataFrame()
# result_with_zeros = pd.DataFrame()
# for f in functions:
#     temp = []
#     temp_with_zeros = []
#     indices = []
#     indices_with_zeros = []
#     for s in studies:
#         print(f'path::{os.path.join(dataset_path, f, s + ".pkl")}')
#         result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f, s + ".pkl"), "rb"))
#
#         arms = list(result_dict.keys())
#         arms.sort()
#         evolution_patients = set()  # Keep track of patients that are already in 'Evolution' trend
#         for arm in arms:
#             # First, gather all patient IDs that are in 'Evolution' trend
#             if 'Evolution' in result_dict[arm]:
#                 evolution_patients.update(result_dict[arm]['Evolution']['patientID'])
#
#             for trend in trends:
#                 index_name = arm + '_' + trend
#                 indices_with_zeros.append(index_name)
#
#                 print(f"Index_name:{index_name}")
#
#                 rSquare_values = result_dict[arm][trend]['rSquare']
#                 patient_ids = result_dict[arm][trend]['patientID']
#
#                 # Check if rSquare_values is empty
#                 if not rSquare_values:
#                     print("rSquare values is empty for arm:", arm, ", trend:", trend)
#                     temp_with_zeros.append(0.0)
#                     continue
#
#                 # Exclude rSquare_values of patients that are in 'Evolution' trend when processing other trends
#                 if trend != 'Evolution':
#                     rSquare_values = [value for i, value in enumerate(rSquare_values) if patient_ids[i] not in evolution_patients]
#
#                 print("rSquare values:", rSquare_values)
#                 temp.append(np.around(np.nanmean(rSquare_values), 3))
#                 temp_with_zeros.append(np.around(np.nanmean(rSquare_values), 3))
#                 indices.append(index_name)
#
#     result[f] = temp
#     result_with_zeros[f] = temp_with_zeros
#



#
# print(result[arms])
# print(result_dict['MPDL3280A_1']['Up']['rSquare'])


# Debugging prints
print("Hello World")  # Indicates new for loop runs
print(f"result.index: {result.index}")
print(f"indices: {indices}")
print(f"temp: {temp}")
print(f"length result: {len(result)}, length indices: {len(indices)}")

result.index = indices
result_with_zeros.index = indices_with_zeros
result.dropna(inplace=True)

#Plotting code
# #colour and Font
# cmap = sns.cm.rocket
# mpl.rcParams['font.size'] = 10
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
#
# # Generate heatmap for result DataFrame
# plt.figure()
# plt.tight_layout()
# tab_n_result = result.div(result.max(axis=1), axis=0)
# ax = sns.heatmap(tab_n_result, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True)
# ax.set_xticklabels(labels=functions, rotation=30, fontsize=10)
# plt.title('R-Squared Values Excluding Empty Entries for each arms', fontsize=20)
# tab_n = result.div(result.max(axis=1), axis=0)
# plt.show()
#
# # Generate heatmap for result_with_zeros DataFrame
# plt.figure()
# plt.tight_layout()
# tab_n_result_with_zeros = result_with_zeros.div(result_with_zeros.max(axis=1), axis=0)
# ax = sns.heatmap(tab_n_result_with_zeros, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True)
# ax.set_xticklabels(labels=functions, rotation=30, fontsize=10)
# plt.title('R-Squared Values Including Zeros for Empty Entries for each arms', fontsize=20)
# plt.show()

# Setup style parameters
cmap = sns.cm.rocket
mpl.rcParams['font.size'] = 8
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Create a figure with 2 subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size to avoid overlapping text
plt.tight_layout(pad=5)  # Increase padding between subplots

# Generate heatmap for result DataFrame
tab_n_result = result.div(result.max(axis=1), axis=0)
sns.heatmap(tab_n_result, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True, ax=axs[0])
axs[0].set_xticklabels(labels=functions, rotation=30, fontsize=10)
axs[0].set_title('R-Squared Values Excluding Empty Entries', fontsize=20)

# Generate heatmap for result_with_zeros DataFrame
tab_n_result_with_zeros = result_with_zeros.div(result_with_zeros.max(axis=1), axis=0)
sns.heatmap(tab_n_result_with_zeros, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True, ax=axs[1])
axs[1].set_xticklabels(labels=functions, rotation=30, fontsize=10)
axs[1].set_title('R-Squared Values Including Zeros for Empty Entries', fontsize=20)

# Show both plots in the same window
plt.show()

rSquare_df = pd.DataFrame(result.values, index=indices, columns=functions)
print(rSquare_df)

#Debugging
print("plotting was success!!")
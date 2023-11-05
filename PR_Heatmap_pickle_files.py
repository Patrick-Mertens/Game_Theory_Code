##Adding libraries
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


#Special


import ast
import csv
##Path
#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)



####TESTING GROUND | Functions



#Self made function for the result_dictonary manipulation.
    #C, currently, the patient with evolution also have fluctate results, thus leading to inflating numbers

# def get_excluded_patient_ids(filename):
#     excluded_ids = []
#     with open(filename, 'r') as file:
#         # Skip the header
#         next(file)
#
#         for line in file:
#             # Remove whitespace and check if the line is not empty
#             line = line.strip()
#             if line:
#                 # Convert the string to a list
#                 data = ast.literal_eval(line)
#                 # Extract the patient ID (which is the second element in the list)
#                 patient_id = str(data[1])  # Convert to string
#                 excluded_ids.append(patient_id)
#     return excluded_ids

# def get_excluded_patient_ids(filename):
#     excluded_ids = []
#
#     with open(filename, 'r') as file:
#         for line in file:
#             # Check if line starts with the pattern `['...'],`
#             if line.strip().startswith("['"):
#                 parts = line.split(',')
#                 # PatientID seems to be the second value after each `['...'],` pattern
#                 if len(parts) > 1 | str(parts):
#                     patient_id = parts[1].strip()
#                     excluded_ids.append(patient_id)
#     return excluded_ids

def get_patient_ids_from_trend(trend_data):
    """Extract patient IDs from the trend data"""
    return trend_data.get('patientID', [])


# Define a path to the sample .txt file
sample_file_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab\Error_Params_diff_seed_23_attempted_2_night_Run\1\Exponential\MPDL3280A_1\Error_paramsExponential_1_MPDL3280A_1_Evolution_results.txt"

# # Fetch the list of excluded patient IDs using the function
# excluded_ids = get_excluded_patient_ids(sample_file_path)

# # Print the results
# print("Excluded Patient IDs:")
# for patient_id in excluded_ids:
#     print(patient_id)


#########Start script


##initial condtions, same as VG_nsclc_paper.py,
studies = ['a', 'a', 'c', 'd', 'e']
studies =['1', '2', '3', '4', '5']

trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
#C, this got in Narmian code overwritten almost immediatly, dont know exactly why.
functions =['Gompertz']

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





#C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab\Error_Params_diff_seed_23_attempted_2_night_Run

# #C,this for rSquared
# result = pd.DataFrame()
# result_with_zeros = pd.DataFrame()
#
# for f in functions:
#     temp = []
#     temp_with_zeros = []
#     indices = []
#     indices_with_zeros = []
#
#
#     for s in studies:
#         # Use os.path.join to make the code system-independent
#         #print(f'path::{os.path.join(dataset_path, f,"Seed_24", s + ".pkl")}')
#         result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f,'Gompertz_Edit', s + ".pkl"), "rb"))
#
#         # C, let make indices, specially for flucate and evolution
#         indices_evolution = []
#         indices_flucate = []
#
#         arms = list(result_dict.keys())
#         arms.sort()
#         for arm in arms:
#             # Get patient IDs from the 'Evolution' trend
#             #patient_ids_evolution = get_patient_ids_from_trend(result_dict[arm]['Evolution'])
#
#             #print(f"Patient IDs for arm {arm} and trend evolution: {patient_ids}")
#
#             for trend in trends:
#                 index_name = arm + '_' + trend
#                 indices_with_zeros.append(index_name)
#
#                 print(result_dict[arm][trend]['prediction'])
#
#                 #Removing flucate pateints
#                 if trend == 'Fluctuate':
#                     # Get patient IDs from the 'Evolution' trend
#                     patient_ids_evolution = get_patient_ids_from_trend(result_dict[arm]['Evolution'])
#
#                     # Iterate through the patient IDs and get their index in the 'Fluctuate' trend's patient IDs
#                     indices_to_remove = [result_dict[arm][trend]['patientID'].index(pid) for pid in
#                                          patient_ids_evolution if pid in result_dict[arm][trend]['patientID']]
#
#                     # Sort the indices
#                     indices_to_remove = sorted(indices_to_remove)
#
#                     # Store removed values for verification
#                     removed_rSquare_values = []
#
#                     # Remove elements from 'rSquare' using the found indices
#                     for idx in reversed(indices_to_remove):  # Iterate in reverse to ensure we delete from the end first
#                         removed_rSquare_values.append(result_dict[arm][trend]['rSquare'][idx])
#                         del result_dict[arm][trend]['rSquare'][idx]
#
#
#                 print(f"Index_name:{index_name}")
#
#                 rSquare_values = result_dict[arm][trend]['rSquare']
#
#                 ##Check if aic is empty
#                 # if ....
#                 # aic_values = result_dict[arm][trend]['aic']
#                 #print(f"result_dict:{result_dict[arm]}")
#                 #TIME.sleep(10)
#
#                 # Check if rSquare_values is empty
#                 if not rSquare_values:
#                     print("rSquare values is empty for arm:", arm, ", trend:", trend)
#                     temp_with_zeros.append(0.0)
#                     continue
#
#                 print("rSquare values:", rSquare_values)
#                 temp.append(np.around(np.nanmean(rSquare_values), 6))
#                 temp_with_zeros.append(np.around(np.nanmean(rSquare_values), 6))
#                 indices.append(index_name)
#
#     result[f] = temp
#     result_with_zeros[f] = temp_with_zeros
#
# print(result)
#
#
# ###
# result.index = indices
# result_with_zeros.index = indices_with_zeros
# result.dropna(inplace=True)
#
# ###
# # Setup style parameters
# cmap = sns.cm.rocket
# mpl.rcParams['font.size'] = 8
# plt.rcParams["font.weight"] = "bold"
# plt.rcParams["axes.labelweight"] = "bold"
#
# # Create a figure with 2 subplots
# fig, axs = plt.subplots(1, 2, figsize=(20, 10))  # Adjust the size to avoid overlapping text
# plt.tight_layout(pad=5)  # Increase padding between subplots
#
# # Generate heatmap for result DataFrame
# tab_n_result = result.div(result.max(axis=1), axis=0)
# sns.heatmap(tab_n_result, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True, ax=axs[0])
# axs[0].set_xticklabels(labels=functions, rotation=30, fontsize=10)
# axs[0].set_title('R-Squared Values Excluding Empty Entries', fontsize=20)
#
# # Generate heatmap for result_with_zeros DataFrame
# tab_n_result_with_zeros = result_with_zeros.div(result_with_zeros.max(axis=1), axis=0)
# sns.heatmap(tab_n_result_with_zeros, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True, ax=axs[1])
# axs[1].set_xticklabels(labels=functions, rotation=30, fontsize=10)
# axs[1].set_title('R-Squared Values Including Zeros for Empty Entries', fontsize=20)
#
# # Show both plots in the same window
# plt.show()
#
# rSquare_df = pd.DataFrame(result.values, index=indices, columns=functions)
# print(rSquare_df)
#
# #Debugging
# print("plotting was success!!")
#


# #####Editing the above code to also handle aic

# Initialize DataFrames to store results for rSquare and aic separately
result_rSquare = pd.DataFrame()
result_aic = pd.DataFrame()

# Initialize DataFrames to store results with zeros for rSquare and aic separately
result_with_zeros_rSquare = pd.DataFrame()
result_with_zeros_aic = pd.DataFrame()

# Loop over each function to compute values
for f in functions:
    # Temporary lists to accumulate results for each function
    temp_rSquare = []
    temp_aic = []

    # Temporary lists to accumulate results (including zeros for empty datasets) for each function
    temp_with_zeros_rSquare = []
    temp_with_zeros_aic = []

    # Lists to hold the indices for results (excluding zeros for empty datasets)
    indices_rSquare = []
    indices_aic = []

    # Lists to hold all the indices, including those with zeros
    indices_with_zeros_rSquare = []
    indices_with_zeros_aic = []

    # Loop over each study
    for s in studies:
        # Load the data from the pickle file
        result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f, s + ".pkl"), "rb"))

        # Sorting the arm keys for consistency
        arms = list(result_dict.keys())
        arms.sort()

        # Loop over each arm
        for arm in arms:
            # Loop over each trend
            for trend in trends:
                # Create a composite index name from arm and trend
                index_name = arm + '_' + trend

                # Handling rSquare values
                # If the trend is 'Fluctuate', remove patients from the 'Evolution' trend
                if trend == 'Fluctuate':
                    # Print rSquare array length before removal
                    print(
                        f"rSquare array length for {arm} and {trend} before removal: {len(result_dict[arm][trend]['rSquare'])}")

                    # Fetch patient IDs associated with the 'Evolution' trend
                    patient_ids_evolution = get_patient_ids_from_trend(result_dict[arm]['Evolution'])

                    # Identify indices of these patients within the 'Fluctuate' trend
                    indices_to_remove = [result_dict[arm][trend]['patientID'].index(pid) for pid in
                                         patient_ids_evolution if pid in result_dict[arm][trend]['patientID']]
                    indices_to_remove.sort()

                    # Remove corresponding rSquare values using the found indices
                    for idx in reversed(indices_to_remove):
                        del result_dict[arm][trend]['rSquare'][idx]

                    # Print rSquare array length after removal
                    print(
                        f"rSquare array length for {arm} and {trend} after removal: {len(result_dict[arm][trend]['rSquare'])}")

                # Fetch rSquare values and handle cases where it's empty
                rSquare_values = result_dict[arm][trend]['rSquare']
                if not rSquare_values:
                    temp_with_zeros_rSquare.append(0.0)
                else:
                    temp_rSquare.append(np.around(np.nanmean(rSquare_values), 3))
                    temp_with_zeros_rSquare.append(np.around(np.nanmean(rSquare_values), 3))
                    indices_rSquare.append(index_name)

                # Handling aic values
                # Fetch aic values and handle cases where it's empty or NaN
                aic_values = result_dict[arm][trend]['aic']
                if not aic_values or all(np.isnan(val) for val in aic_values):
                    temp_with_zeros_aic.append(0.0)
                else:
                    temp_aic.append(np.around(np.nanmean(aic_values), 3))
                    temp_with_zeros_aic.append(np.around(np.nanmean(aic_values), 3))
                    indices_aic.append(index_name)

                # Store the composite index names
                indices_with_zeros_rSquare.append(index_name)
                indices_with_zeros_aic.append(index_name)

    # Store accumulated results in respective DataFrames
    result_rSquare[f] = temp_rSquare
    result_with_zeros_rSquare[f] = temp_with_zeros_rSquare
    result_aic[f] = temp_aic
    result_with_zeros_aic[f] = temp_with_zeros_aic

# Display results for rSquare and aic
print("rSquare Results:")
print(result_rSquare)
print("\nAIC Results:")
print(result_aic)
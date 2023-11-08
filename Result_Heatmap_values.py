##The script, is to obtain the results from the pickple files
#For some reason the fitfunction script can give different result when it run, for this reason only the values are selcted that make sense
#For some runs, the fitfunction results became negative (Rsquared), while other runs gave a more reasonable result

#Packages
import sys
import pickle
import os
import builtins
import numpy as np
import pandas as pd

#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)

#Initial values
studies =['1', '2', '3', '4'] #Study 5 isnt  included,because different type of cancer
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']

#Setting dataframes up
result = pd.DataFrame()
result_with_zeros = pd.DataFrame()

# ##For loop
# for f in functions:
#     temp = []
#     temp_with_zeros = []
#     indices = []
#     indices_with_zeros = []
#     for s in studies:
#         # Use os.path.join to make the code system-independent
#         print(f'path::{os.path.join(dataset_path, f,"Runner_script/seed_43/for_loop_itteration_1/Combo_formula", s + ".pkl")}')
#         result_dict = pickle.load(builtins.open(os.path.join(dataset_path, f,"Runner_script/seed_42/for_loop_itteration_2/Combo_formula", s + ".pkl"), "rb"))
#
#         arms = list(result_dict.keys())
#         arms.sort()
#         for arm in arms:
#             for trend in trends:
#                 index_name = arm + '_' + trend
#                 indices_with_zeros.append(index_name)
#
#                 print(f"Index_name:{index_name}")
#
#                 rSquare_values = result_dict[arm][trend]['rSquare']
#                 aic_values = result_dict[arm][trend]['aic']
#                 rmse_values = result_dict[arm][trend]['rmse']
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
#                 temp.append(np.around(np.nanmean(rSquare_values), 3))
#                 temp_with_zeros.append(np.around(np.nanmean(rSquare_values), 3))
#                 indices.append(index_name)
#
#     result[f] = temp
#     result_with_zeros[f] = temp_with_zeros


##### EDITED THE ABOVE FOR LOOP IN MORE DETAIL, ADDED PATIENTID TO DOUBLE CHECK THE VALUES
# # Initialize DataFrame to store patientID lengths
# patientID_lengths_df = pd.DataFrame(columns=['arm', 'trend', 'patientID_length'])

# Initialize a list to store your data
data_list = []

# Main processing loop
# for f in functions:
#     temp_rSquare = []
#     temp_rSquare_with_zeros = []
#     temp_aic = []
#     temp_aic_with_zeros = []
#     temp_rmse = []
#     temp_rmse_with_zeros = []
#
#     indices = []
#     indices_with_zeros = []
#
#     for s in studies:
#         # Use os.path.join to make the code system-independent
#         print(
#             f'path::{os.path.join(dataset_path, f, s + ".pkl")}')
#         result_dict = pickle.load(builtins.open(
#             os.path.join(dataset_path, f,  s + ".pkl"),
#             "rb"))
#
#         arms = list(result_dict.keys())
#         arms.sort()
#         for arm in arms:
#             for trend in trends:
#                 index_name = arm + '_' + trend
#                 indices_with_zeros.append(index_name)
#
#                 print(f"Index_name:{index_name}")
#
#                 rSquare_values = result_dict[arm][trend]['rSquare']
#                 aic_values = result_dict[arm][trend]['aic']
#                 rmse_values = result_dict[arm][trend]['rmse']
#                 patientID = result_dict[arm][trend].get('patientID', [])
#
#                 # Appending a new row to the DataFrame
#                 # new_row = {'arm': arm, 'trend': trend, 'patientID_length': len(patientID)}
#                 # patientID_lengths_df = patientID_lengths_df.pd.append(new_row, ignore_index=True)
#                 # Collecting patientID length data
#                 data_list.append({
#                     'arm': arm,
#                     'trend': trend,
#                     'patientID_length': len(patientID)
#                 })
#
#
#                 # Check if rSquare_values is empty
#                 if not rSquare_values:
#                     print("rSquare values is empty for arm:", arm, ", trend:", trend)
#                     temp_rSquare_with_zeros.append(0.0)
#                 else:
#                     temp_rSquare.append(np.around(np.nanmean(rSquare_values), 3))
#                     temp_rSquare_with_zeros.append(np.around(np.nanmean(rSquare_values), 3))
#                     indices.append(index_name)
#
#                 # Check if aic_values is empty
#                 if not aic_values:
#                     print("AIC values is empty for arm:", arm, ", trend:", trend)
#                     temp_aic_with_zeros.append('empty')
#                 else:
#                     temp_aic.append(np.around(np.nanmean(aic_values), 3))
#                     temp_aic_with_zeros.append(np.around(np.nanmean(aic_values), 3))
#
#                 # RMSE values processing
#                 if not rmse_values:
#                     print("RMSE values is empty for arm:", arm, ", trend:", trend)
#                     temp_rmse_with_zeros.append(0.0)
#                 else:
#                     temp_rmse.append(np.around(np.nanmean(rmse_values), 3))
#                     temp_rmse_with_zeros.append(np.around(np.nanmean(rmse_values), 3))
#
#     # Now temp lists contain data for all studies. You can process them further as needed.
#
#
# patientID_lengths_df = pd.DataFrame(data_list)
# # After the loop, the patientID_lengths_df DataFrame contains the length of patientID for each arm and trend
# print(patientID_lengths_df)

###Small tweak, instead of dataframe, a list first and than transformed to dataframe

for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path, f, s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path, f,  s + ".pkl"),
            "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                # Get rSquare, aic, and rmse values
                rSquare_value = np.around(np.nanmean(result_dict[arm][trend].get('rSquare', [0.0])), 3)
                aic_value = np.around(np.nanmean(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
                    trend].get('aic') else 'empty'
                rmse_value = np.around(np.nanmean(result_dict[arm][trend].get('rmse', [0.0])), 12)
                print(rmse_value)
                patientID = result_dict[arm][trend].get('patientID', [])

                # Collecting data
                data_list.append({
                    'fit_function': f,
                    'arm': arm,
                    'trend': trend,
                    'patientID_length': len(patientID),
                    'rSquare': rSquare_value,
                    'aic': aic_value,
                    'rmse': rmse_value
                })

#Debuging:
print(result_dict)

#Transformring it to a dataframe
df = pd.DataFrame(data_list)

#Check
total_patients = df['patientID_length'].sum()
print("Total number of patients:", total_patients)
#Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#Printing
print(df)

Set1 = df


##Another set

data_list = []

for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path, f,"Changed_formula", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path, f,  s + ".pkl"),
            "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                # Get rSquare, aic, and rmse values
                rSquare_value = np.around(np.nanmean(result_dict[arm][trend].get('rSquare', [0.0])), 3)
                aic_value = np.around(np.nanmean(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
                    trend].get('aic') else 'empty'
                rmse_value = np.around(np.nanmean(result_dict[arm][trend].get('rmse', [0.0])), 12)
                print(rmse_value)
                patientID = result_dict[arm][trend].get('patientID', [])

                # Collecting data
                data_list.append({
                    'fit_function': f,
                    'arm': arm,
                    'trend': trend,
                    'patientID_length': len(patientID),
                    'rSquare': rSquare_value,
                    'aic': aic_value,
                    'rmse': rmse_value
                })

#Debuging:
print(result_dict)

#Transformring it to a dataframe
df = pd.DataFrame(data_list)

#Check
total_patients = df['patientID_length'].sum()
print("Total number of patients:", total_patients)
#Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#Printing
print(df)

Set2 = df

##Another set

data_list = []

for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path, f,"Runner_script/seed_42/for_loop_itteration_1/Combo_formula", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path, f,  s + ".pkl"),
            "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                # Get rSquare, aic, and rmse values
                rSquare_value = np.around(np.nanmean(result_dict[arm][trend].get('rSquare', [0.0])), 3)
                aic_value = np.around(np.nanmean(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
                    trend].get('aic') else 'empty'
                rmse_value = np.around(np.nanmean(result_dict[arm][trend].get('rmse', [0.0])), 12)
                print(rmse_value)
                patientID = result_dict[arm][trend].get('patientID', [])

                # Collecting data
                data_list.append({
                    'fit_function': f,
                    'arm': arm,
                    'trend': trend,
                    'patientID_length': len(patientID),
                    'rSquare': rSquare_value,
                    'aic': aic_value,
                    'rmse': rmse_value
                })

#Debuging:
print(result_dict)

#Transformring it to a dataframe
df = pd.DataFrame(data_list)

#Check
total_patients = df['patientID_length'].sum()
print("Total number of patients:", total_patients)
#Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#Printing
print(df)

Set3 = df


###Another Set

data_list = []

for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path, f,"Runner_script_study1to4/seed_42/for_loop_itteration_1/Combo_formula", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path, f,  s + ".pkl"),
            "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                # Get rSquare, aic, and rmse values
                rSquare_value = np.around(np.nanmean(result_dict[arm][trend].get('rSquare', [0.0])), 3)
                aic_value = np.around(np.nanmean(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
                    trend].get('aic') else 'empty'
                rmse_value = np.around(np.nanmean(result_dict[arm][trend].get('rmse', [0.0])), 12)
                print(rmse_value)
                patientID = result_dict[arm][trend].get('patientID', [])

                # Collecting data
                data_list.append({
                    'fit_function': f,
                    'arm': arm,
                    'trend': trend,
                    'patientID_length': len(patientID),
                    'rSquare': rSquare_value,
                    'aic': aic_value,
                    'rmse': rmse_value
                })

#Debuging:
print(result_dict)

#Transformring it to a dataframe
df = pd.DataFrame(data_list)

#Check
total_patients = df['patientID_length'].sum()
print("Total number of patients:", total_patients)
#Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
#Printing
print(df)

Set4 = df

###Making it an Excel File:
# Create a Pandas Excel writer using XlsxWriter as the engine.
excel_file_path = r"C:\Users\Shade\Desktop\Fitting_results_combined_data.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    Set1.to_excel(writer, sheet_name='Set1', index=False)
    Set2.to_excel(writer, sheet_name='Set2', index=False)
    Set3.to_excel(writer, sheet_name='Set3', index=False)
    Set4.to_excel(writer, sheet_name='Set4', index=False)
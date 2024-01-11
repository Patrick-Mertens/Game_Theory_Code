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


# Initialize a list to store your data
data_list = []


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
                aic_value = np.around(np.nanmedian(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
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
                aic_value = np.around(np.nanmedian(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
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
                aic_value = np.around(np.nanmedian(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
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
                aic_value = np.around(np.nanmedian(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
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
excel_file_path = r"C:\Users\Shade\Desktop\Fitting_results_combined_data_aic_median.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    Set1.to_excel(writer, sheet_name='Set1', index=False)
    Set2.to_excel(writer, sheet_name='Set2', index=False)
    Set3.to_excel(writer, sheet_name='Set3', index=False)
    Set4.to_excel(writer, sheet_name='Set4', index=False)
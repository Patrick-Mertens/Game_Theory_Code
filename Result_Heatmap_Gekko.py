#Packages
import sys
import pickle
import os
import builtins
import numpy as np
import pandas as pd

#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"/home/bob/project_game_theory_linux/SpiderProject_KatherLab"
sys.path.insert(0,dataset_path)
#C:\Users\Shade\Desktop\Master\Project Game Theory Code\5_nov\linux_scripts\project_game_theory_linux\SpiderProject_KatherLab\Results_nsclc_paper_3\PR\1\Sigma 0\Exponential\Evolution\Pickle_Files

#Initial values
studies =['1', '2', '3', '4'] #Study 5 isnt  included,because different type of cancer
functions = ['Exponential']
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']

# Initialize a list to store your data
data_list = []

#
for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path,"PR", s,"Sigma 0","Exponential","Evolution","Pickle_Files", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path,"PR", s,"Sigma 0","Exponential","Evolution","Pickle_Files", s + ".pkl"),
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
#print(result_dict)

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
#REMEMBER, FLUCTATE CONTAIN EVOLUTION!!!


# Initialize a list to store your data
data_list = []

#
for f in functions:


    for s in studies:
        # Use os.path.join to make the code system-independent
        print(
            f'path::{os.path.join(dataset_path,"VG", s,"Sigma 0","Exponential","Evolution","Pickle_Files", s + ".pkl")}')
        result_dict = pickle.load(builtins.open(
            os.path.join(dataset_path,"VG", s,"Sigma 0","Exponential","Evolution","Pickle_Files", s + ".pkl"),
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
#print(result_dict)

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
#REMEMBER, FLUCTATE CONTAIN EVOLUTION!!!

excel_file_path = r"C:\Users\Shade\Desktop\GEKKO_rSquared_combined_data.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    Set1.to_excel(writer, sheet_name='PR', index=False)
    Set2.to_excel(writer, sheet_name='VG_Split_3', index=False)

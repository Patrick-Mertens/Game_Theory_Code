#Packages
import sys
import pickle
import os
import builtins
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\5_nov\linux_scripts\project_game_theory_linux\SpiderProject_KatherLab\Results_nsclc_paper_3"  # Use a raw string for the path
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

                #Debugging
                # print(f"PatientID, arm {arm}, trend: {trend} PatientID: {result_dict[arm][trend]['patientID']}")
                # print(f"ModelPrediction: arm {arm}, trend: {trend} Prediction: {result_dict[arm][trend]['prediction']}")
                # print(f"Dimension:arm {arm}, trend: {trend}, Dimension: {result_dict[arm][trend]['dimension']}")
                # print(f"Time:arm {arm}, trend: {trend}: time{result_dict[arm][trend]['dimension']} ")
                # print(f"Rsquared, trend:{trend},{arm}, rSquared:{result_dict[arm][trend].get('rSquare', [0.0])}")
                # print(f"Dimension First: {result_dict[arm][trend]['dimension'][0]}")
                # print(f"Prediction First:{result_dict[arm][trend]['prediction'][0]}")
                # print(f"rSquared caculated {trend},{arm}, rSquared:{r2_score(result_dict[arm][trend]['prediction'][0],result_dict[arm][trend]['dimension'][0])}")
                # print(f"AIC arm for trend:{trend},{arm}, aic:{result_dict[arm][trend].get('aic', [0.0])}")
                aic_value = np.around(np.nanmedian(result_dict[arm][trend].get('aic', [0.0])), 3) if result_dict[arm][
                    trend].get('aic') else 'empty'
                rmse_value = np.around(np.nanmean(result_dict[arm][trend].get('rmse', [0.0])), 12)

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



####If you are searching for a specific patient Prediction
# import matplotlib.pyplot as plt
# import os
# import pickle
#
# patient_id_to_find = "XXXXXX"
# found_data = False
#
#
# for s in studies:
#     result_dict = pickle.load(open(
#         os.path.join(dataset_path, "PR", s, "Sigma 0", "Exponential", "Evolution", "Pickle_Files", s + ".pkl"),
#         "rb"))
#
#     arms = list(result_dict.keys())
#     arms.sort()
#     for arm in arms:
#         for trend in trends:
#             patientID_list = result_dict[arm][trend].get('patientID', [])
#             print(patientID_list)
#             if patient_id_to_find in patientID_list:
#                 # Find the index of the patient ID
#                 patient_index = patientID_list.index(patient_id_to_find)
#                 #print(f"index:{patient_index}")
#                 # Extract the data for the found patient index
#                 prediction_to_plot = result_dict[arm][trend]['prediction'][patient_index]
#                 print(f"Prediction: {prediction_to_plot}")
#                 time_to_plot = result_dict[arm][trend]['time'][patient_index]  # Assuming this is the correct key
#                 print(f"time: {time_to_plot}")
#                 dimension_to_plot = result_dict[arm][trend]['dimension'][patient_index]
#                 print(f"dimension_to_plot")
#                 found_data = True
#                 break  # Break out of the innermost loop
#         if found_data:
#             break  # Break out of the middle loop
#     if found_data:
#         break  # Break out of the outermost loop
#
# # Now plot the data if found
# if found_data:
#     plt.figure(figsize=(10, 5))
#     plt.plot(time_to_plot, prediction_to_plot, label='model predictions', color='blue')
#     plt.scatter(time_to_plot, dimension_to_plot, label='real measurements', color='red')
#     plt.xlabel('Time')
#     plt.ylabel('Dimension/Prediction')
#     plt.title(f'Patient {patient_id_to_find} Data')
#     plt.legend()
#     plt.show()
# else:
#     print(f"No data found for patient {patient_id_to_find}.")




excel_file_path = r"C:\Users\Shade\Desktop\GEKKO_rSquared_combined_data_aic_median.xlsx"
with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
    Set1.to_excel(writer, sheet_name='PR', index=False)
    Set2.to_excel(writer, sheet_name='VG', index=False) #C, TO see if different params for different environement

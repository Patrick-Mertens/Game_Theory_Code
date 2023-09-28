import numpy as np
from scipy.optimize import curve_fit
import warnings
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import Utils as utils
import FitFunctions as ff
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Normalizer
import matplotlib as mpl
from matplotlib.lines import Line2D
import sys
import builtins
import csv

#Importing the local functions
from VG_Functions_2 import *


#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited"  # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "Control_20_09_2023_attempt_7_redrawing" #C, to many patients


#Other stuff
# Read the CSV file to get the list of patient IDs
csv_file_path = os.path.join(dataset_path,"PatientID", 'df_Size1_and_Increase.CSV')
df_from_csv = pd.read_csv(csv_file_path)
patientIDs_from_csv = df_from_csv['PatientID'].tolist()
chemo =['DOCETAXEL','docetaxel']

#Printing
print(f"patientIDs_from_csv: {patientIDs_from_csv}")


#Initial settings
studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential']#, 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies =['1', '2', '3', '4', '5']


# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []

for studyName in studies:
    #rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path, 'SpiderProject_KatherLab', 'Study_' + studyName + '_1.xlsx')
    sind = studies.index(studyName)
    sp = splits[sind]
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']   #take only first tumor
    filtered_Data = filtered_Data.loc[filtered_Data['TRTESTCD'] == 'LDIAM'] #take only tumors for which measurement of longer diameter is available
    temp = list(filtered_Data['TRORRES'])  #this should be the measurements
    temp = utils.Remove_String_From_Numeric_Vector(temp, valueToReplace = 0) #removes strings and replace by zero, why? do we only have strings when it disappears?
    #temp = transform_to_volume(temp)
    maxList.append(max(temp)) #max value of measurement
    minList.append(min(temp))    #min value of measurement


####THE ABOVE CODE WAS FROM ANOTHER BLOCK BUT IT LOOKS EXACTLY THE SAME ####


# Fit Funtions to the Data Points

#maxi = np.max([288, 0])
maxi = np.max(maxList)
scaled_days=[]
scaled_pop=[]
list_arms=[]
list_study=[]
list_patients=[]
studies = ['a', 'a', 'c', 'd', 'e']
studies =['1', '2', '3', '4']
#studies=['3']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
functions =['Exponential']
splits = [True, True, False, True, True]
noPars = [3, 3, 3, 4, 3, 4]
list_trends=[]
#splits =[False] #only for study 3
count_d=0

for studyName in studies:
    scaled_days_i=[]
    scaled_pop_i =[]
    sind = studies.index(studyName)
    sp = splits[sind]
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True

    #rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path,'SpiderProject_KatherLab',  'Study_' + studyName + '_1.xlsx')
    data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
    for functionToFit in functions:

        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = utils.Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate', 'Evolution'], categories = ['patientID', 'rmse', 'rSquare',
                                                                                                'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
        print(functionToFit)
        print(studyName)

        for arm in arms:

            print(arm)

            data_temp = data.loc[data['receivedTreatment'] == arm]
            patientID = list(data_temp['USUBJID'].unique())

            for key in patientID:

                filteredData = data.loc[data['USUBJID'] == key]
                temp = filteredData['TRLINKID'].unique()
                temp = [i for i in temp if not str(i) == 'nan']
                temp = [i for i in temp if not '-NT' in str(i)]

                if 'INV-T001' in temp:
                    tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                    tumorFiltered_Data.dropna(subset=['TRDY'], inplace=True)
                    # tumorFiltered_Data.dropna(subset = ['VISITDY'], inplace = True)

                    tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']

                    # Limit the Data Points for 6 and bigger!
                    keysList = []
                    if len(tumorFiltered_Data) >= 6:
                        dimension = list(tumorFiltered_Data['TRORRES'])
                        time = list(tumorFiltered_Data['TRDY'])
                        # time = list(tumorFiltered_Data['VISITDY'])

                        time = utils.Correct_Time_Vector(time, convertToWeek=True)

                        # If the value of Dimension is nan or any other string value, we replace it with zero
                        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace=0)

                        dimension = [x for _, x in sorted(zip(time, dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy / maxi
                            # dimension_copy = dimension_copy/np.max(dimension_copy)

                        trend = utils.Detect_Trend_Of_Data(dimension_copy)

                        dimension = [i * i * i * 0.52 for i in dimension]  # transform to volume
                        if normalizeDimension:
                            dimension = dimension / np.max([maxi * maxi * maxi * 0.52, 0])  #
                        time.sort()
                        cn = list(tumorFiltered_Data['TULOC'])[0]

                        # scale my way
                        # dimension = transform_to_volume(dimension)
                        # dimension = scale_data(dimension, maxi)
                        scaled_days.append(time)
                        scaled_pop.append(dimension)
                        list_trends.append(trend)
                        list_arms.append(arm)
                        list_study.append(studyName)
                        list_patients.append(key)

                        print(f" 1 list_patients: {len(list_patients)}")

                        '''
                        try:
                             modelPredictions = fitfunc(time, *fittedParameters)

                        except:
                                print(key)
                                result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'],
                                                                          values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn])
                                continue

                        if len(set(dimension)) == 1:
                            modelPredictions = dimension
                        else:
                            modelPredictions =

                        modelPredictions = [0 if str(i) == 'nan' else i  for i in modelPredictions]
                        absError = modelPredictions - dimension
                        SE = np.square(absError)
                        temp_sum = np.sum(SE)
                        MSE = np.mean(SE)

                        result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'],
                                                                          values = [key, time, dimension, modelPredictions, mean_squared_error(dimension, modelPredictions),
                                                                                    r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), fittedParameters, cn])


        #a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        a_file = open(os.path.join(dataset_path, functionToFit, studyName + '.pkl'), "wb")

        pickle.dump(result_dict, a_file)
        a_file.close()'''
    # scaled_days.append(scaled_days_i)
    # scaled_pop.append(scaled_pop_i)
lim = limit(scaled_pop)
# Size1, Size2, Size4, Inc, Dec = split1(lim/2, lim*2, scaled_pop)
Size1, Size2, Size3, Size4, Inc, Dec = split1(lim / 20, lim / 2, lim * 3, scaled_pop)  # C, I think these need to be saved
# Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim / 20,
#                                                                                   lim / 2,
#                                                                                   lim * 3,
#                                                                                   dimension,
#                                                                                   trend)


########################################################################
list_d=[]
list_m=[]
list_m2=[]
for i in range(len(list_arms)):
  if 'DOCETAXEL' in list_arms[i] or 'Docetaxel' in list_arms[i]:
    list_d.append((i))
  elif 'MPDL3280A' in list_arms[i]:
    list_m.append((i))
  else:
    list_m2.append((i))


Inc=[]
Dec=[]
for i in range(len(scaled_pop)):
  if i in list_m:
    if scaled_pop[i][0]> scaled_pop[i][1]:
      Dec.append((i))
    else:
      Inc.append((i))


list_trends
Up=[]
Down=[]
Fluctuate=[]
Evolution=[]
for i in range(len(list_trends)):
  #if i in list_d:
    if list_trends[i] == 'Up':
      Up.append((i))
    elif list_trends[i] == 'Down':
      Down.append((i))
    elif list_trends[i] == 'Evolution':
      Evolution.append((i))
    elif list_trends[i] == 'Fluctuate':
      Fluctuate.append((i))


filtered =[]

#Debugging
print(f" 2 list_patients: {len(list_patients)}")
for i in Size2:
  filtered.append(list_patients[i])
print(len(set(filtered)))

print(len(set(list_patients)))

print(len(list_patients)) #C, I need to print this
print(f" 3 list_patients: {len(list_patients)}")

list_3a=[]
list_b=[]
for i in range(len(list_arms)):

  if list_arms[i] == 'Cohort 3a (Squamous)' or list_arms[i] == 'Cohort 2a (Squamous)' or list_arms[i] == 'Cohort 1a (Squamous)':
    list_3a.append((i))
  else:
    list_b.append((i))



# Printing values for debugging with types
print("\n==== DEBUGGING INFORMATION ====")

# Printing the values and types in list_b
print("\n--- list_b ---")
print(list_b)
print(f"Total Count: {len(list_b)}")
print(f"Type: {type(list_b)}")

# Printing the values and types in list_3a
print("\n--- list_3a ---")
print(list_3a)
print(f"Total Count: {len(list_3a)}")
print(f"Type: {type(list_3a)}")

# Printing the values and types in list_d
print("\n--- list_d ---")
print(list_d)
print(f"Total Count: {len(list_d)}")
print(f"Type: {type(list_d)}")

# Printing the values and types in list_m
print("\n--- list_m ---")
print(list_m)
print(f"Total Count: {len(list_m)}")
print(f"Type: {type(list_m)}")

# Printing the values and types in list_m2
print("\n--- list_m2 ---")
print(list_m2)
print(f"Total Count: {len(list_m2)}")
print(f"Type: {type(list_m2)}")

# Printing the values and types in Inc
print("\n--- Inc ---")
print(Inc)
print(f"Total Count: {len(Inc)}")
print(f"Type: {type(Inc)}")

# Printing the values and types in Dec
print("\n--- Dec ---")
print(Dec)
print(f"Total Count: {len(Dec)}")
print(f"Type: {type(Dec)}")

# Printing the values and types in list_trends
print("\n--- list_trends ---")
print(list_trends)
print(f"Total Count: {len(list_trends)}")
print(f"Type: {type(list_trends)}")

# Printing the values and types in Up, Down, Fluctuate, Evolution
print("\n--- Up ---")
print(Up)
print(f"Total Count: {len(Up)}")
print(f"Type: {type(Up)}")

print("\n--- Down ---")
print(Down)
print(f"Total Count: {len(Down)}")
print(f"Type: {type(Down)}")

print("\n--- Fluctuate ---")
print(Fluctuate)
print(f"Total Count: {len(Fluctuate)}")
print(f"Type: {type(Fluctuate)}")

print("\n--- Evolution ---")
print(Evolution)
print(f"Total Count: {len(Evolution)}")
print(f"Type: {type(Evolution)}")

# Printing the values and types in filtered and list_patients
print("\n--- filtered ---")
print(filtered)
print(f"Total Count: {len(filtered)}")
print(f"Type: {type(filtered)}")

print("\n--- list_patients ---")
print(list_patients)
print(f"Total Count: {len(list_patients)}")
print(f"Type: {type(list_patients)}")

# Printing the values and types in scaled_pop and scaled_days
print("\n--- scaled_pop ---")
print(scaled_pop)
print(f"Total Count: {len(scaled_pop)}")
print(f"Type: {type(scaled_pop)}")

print("\n--- scaled_days ---")
print(scaled_days)
print(f"Total Count: {len(scaled_days)}")
print(f"Type: {type(scaled_days)}")

# Printing the values and types in Size1 and Size2
print("\n--- Size1 ---")
print(Size1)
print(f"Total Count: {len(Size1)}")
print(f"Type: {type(Size1)}")

print("\n--- Size2 ---")
print(Size2)
print(f"Total Count: {len(Size2)}")
print(f"Type: {type(Size2)}")

print("\n--- Size3 ---")
print(Size3)
print(f"Total Count: {len(Size3)}")
print(f"Type: {type(Size3)}")

print("\n--- Size4 ---")
print(Size4)
print(f"Total Count: {len(Size4)}")
print(f"Type: {type(Size4)}")

print("\n=== END OF DEBUGGING INFORMATION ===\n")


#Getting param
Test_result = PR_get_param_2(Size1, Inc, scaled_pop, scaled_days, list_d)
#Print
print(f"param: {Test_result}")

# docetaxel_indices = [i for i in range(len(list_arms)) if 'DOCETAXEL' in list_arms[i].upper()]
#
# docetaxel_scaled_pop = []
# docetaxel_scaled_days = []
# docetaxel_inc_indices = []
#
# for i in docetaxel_indices:
#     if i in Inc:
#         docetaxel_scaled_pop.append(scaled_pop[i])
#         docetaxel_scaled_days.append(scaled_days[i])
#         docetaxel_inc_indices.append(i)
#
# # If necessary, apply additional filtering to obtain the Size1 population, using the appropriate condition
# Size1 = [docetaxel_scaled_pop[i] for i in range(len(docetaxel_scaled_pop)) if your_size_condition]  # Replace 'your_size_condition' with the actual condition
#
# # Assuming that the response is another attribute you need to filter out based on the indices in docetaxel_inc_indices
# response = [derive_response_attribute(docetaxel_inc_indices[i]) for i in range(len(docetaxel_inc_indices))]  # Replace with actual attribute extraction
#
# # Now, you should have the filtered populations and days, and can proceed to call the PR_get_param function
# result = PR_get_param(Size1, response, docetaxel_scaled_pop, docetaxel_scaled_days)
#
# print(f"{result}")
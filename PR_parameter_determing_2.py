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
import time as TIME


#Importing the local functions
from VG_Functions_2 import *

#Attempting something
from itertools import combinations
from datetime import datetime

#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "Control_20_09_2023_attempt_7_redrawing" #C, to many patients


#Initial settings
studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential']#, 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies =['1', '2', '3', '4', '5']

###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []

for studyName in studies:
    #rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path,  'Study_' + studyName + '_1.xlsx')
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
#studies=['3'] #C, uncommeted to see what happens
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
    rawDataPath = os.path.join(dataset_path,  'Study_' + studyName + '_1.xlsx')
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

                if  'INV-T001' in temp :
                    tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
                    tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)
                    #tumorFiltered_Data.dropna(subset = ['VISITDY'], inplace = True)

                    tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']

                    # Limit the Data Points for 6 and bigger!
                    keysList = []
                    if len(tumorFiltered_Data) >= 6:
                        dimension = list(tumorFiltered_Data['TRORRES'])
                        time = list(tumorFiltered_Data['TRDY'])
                        #time = list(tumorFiltered_Data['VISITDY'])



                        time = utils.Correct_Time_Vector(time, convertToWeek = True)

                        # If the value of Dimension is nan or any other string value, we replace it with zero
                        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)

                        dimension = [x for _,x in sorted(zip(time,dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy/maxi
                            #dimension_copy = dimension_copy/np.max(dimension_copy)

                        trend = utils.Detect_Trend_Of_Data(dimension_copy)

                        dimension = [i * i * i * 0.52 for i in dimension] #transform to volume
                        if normalizeDimension:
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.52, 0])  #
                        time.sort()
                        cn =   list(tumorFiltered_Data['TULOC']) [0]

                        #scale my way
                        #dimension = transform_to_volume(dimension)
                        #dimension = scale_data(dimension, maxi)
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
        #scaled_days.append(scaled_days_i)
         #scaled_pop.append(scaled_pop_i)
lim = limit(scaled_pop)
#Size1, Size2, Size4, Inc, Dec = split1(lim/2, lim*2, scaled_pop)
Size1, Size2,Size3, Size4, Inc, Dec = split1(lim/20, lim/2, lim*3, scaled_pop) #C, I think these need to be saved

#Printing
print(f"lim: {lim}") #C, the same as the ipyn: 0.003316135059100506
print(f"Size1: {Size1}")
print(f"Size2: {Size2}")
print(f"Size3: {Size3}")
print(f"Size4: {Size4}")
print(f"Inc: {Inc}")
print(f"Dec: {Dec}")


########################################################################
#C, This part of teh code does the filtering, by filterting the inc, decrease array index
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
  if i in list_d: #changed this from list_m to list_d, to get Inc only coresponding to chemotherapy group
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
#Printing coresponding patient ID
for i in Size1:
  filtered.append(list_patients[i])
print(len(set(filtered)))
print(f"Size1: {filtered}")

filtered2 = []

#Printing coresponding patient ID
for i in Inc:
  filtered2.append(list_patients[i])
print(len(set(filtered2)))
print(f"Inc: {filtered2}")

filtered3 = []
#Printing coresponding patient ID
for i in list_d:
  filtered3.append(list_patients[i])
print(len(set(filtered3)))
print(f"List_d: {filtered3}")

print(f"lenght of list arms: {len(list_arms)}")
print(f"lenght of list_d: {len(list_d)}")
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


print("#################################GETTING PARAM#####################")


print(f"scaled_pop: {scaled_pop}")
print(f"length scaled_pop: {len(scaled_pop)}")
print(f"scaled_days: {scaled_days}")
print(f"length scaled_days: {len(scaled_days)}")


target_dir = os.path.join(dataset_path, "Param_size1_control")
if not os.path.exists(target_dir):
    os.makedirs(target_dir)


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

#Common index beteween size1, Inc
print(f"\n--- Size1 Inc ---")
common_index = {370, 359, 335}
print(f"common_index as Checkin union: {common_index}")


for i in common_index:
    print(f"for {i} in list_patients: {list_patients[i]}")


# Convert the lists to sets
Size1_set = set(Size1)
Inc_set = set(Inc)
list_d_set = set(list_d)

# Find the common numbers
common_Size1_Inc = Size1_set.intersection(Inc_set)
common_Size1_list_d = Size1_set.intersection(list_d_set)
common_Inc_list_d = Inc_set.intersection(list_d_set)
common_all_three = Size1_set.intersection(Inc_set, list_d_set)

# Print the common numbers
print("Common between Size1 and Inc:", common_Size1_Inc)
print("Common between Size1 and list_d:", common_Size1_list_d)
print("Common between Inc and list_d:", common_Inc_list_d)
print("Common between all three lists:", common_all_three)


for i in common_all_three:
    print(f"for {i} in list_patients and in {common_all_three}: {list_patients[i]}")
print("\n=== END OF DEBUGGING INFORMATION ===\n")



Run_Total = False
if Run_Total == True:
    #Doing new gridsearch:
    #PR_get_param_Fixed(size, response, scaled_pop, scaled_days, patient_list, Control):
    Result_run_model_fixed_fixed = PR_get_param_Fixed(Size1,Inc,scaled_pop,scaled_days,list_patients,Control=0)
    #printing result
    print(f"Result_run_model_fixed_fixed, Size1 Inc: {Result_run_model_fixed_fixed}")

    TIME.sleep(120)

    ###OLD it works mostly, need to replace common_indexes with, common_all_three
    # #all patients
    # # Call the get_param function and store the result
    result_run_model_fixed = PR_get_param(Size1, Inc, scaled_pop,scaled_days,list_patients) #Modified function, added scaled_pop to it

    print(f"result_run_model_fixed, Size1 Inc: {result_run_model_fixed}")
#

#####Switches#####
Run_Singles = False
#C, adivce do one of the statements, and let it corespond in the datapath
Index_kept = False
Index_remod = False
####################

#Single runes
if Run_Singles == True:



    #Debug funciton
    # def debug_function(a, b, c, d):
    #     print("Modified Size1:", a)
    #     print("Type of Modified Size1:", type(a))
    #     print("Length of Modified Size1:", len(a))
    #     print("--------------------------")
    #
    #     print("Modified Inc:", b)
    #     print("Type of Modified Inc:", type(b))
    #     print("Length of Modified Inc:", len(b))
    #     print("--------------------------")
    #
    #     print("Modified scaled_pop:", c)
    #     print("Type of Modified scaled_pop:", type(c))
    #     print("Length of Modified scaled_pop:", len(c))
    #     for i, array in enumerate(c):
    #         print(f"Array {i} length:", len(array))
    #     print("--------------------------")
    #
    #     print("Modified scaled_days:", d)
    #     print("Type of Modified scaled_days:", type(d))
    #     print("Length of Modified scaled_days:", len(d))
    #     for i, array in enumerate(d):
    #         print(f"Array {i} length:", len(array))
    #     print("--------------------------")
    #
    #
    # Iterate through each index in the common_indexes, index kept, common removed
    #C, this part is commeted at the moment to prevent overflowing of the folder
    #Run_model_fixed_fixed

    if Index_kept == True:

        for index in common_all_three:
            # Only include the current index in the modified_Size1
            modified_Size1 = [x for x in Size1 if x == index]

            # The other lists remain unchanged
            modified_Inc = Inc
            modified_scaled_pop = scaled_pop
            modified_scaled_days = scaled_days
            modified_Patient_list = list_patients

            print(f"Kept index: {index}")
            # debug_function(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list)

            # Assuming you have the actual PR_get_param function, replace the line below
            result = PR_get_param_Fixed(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list,Control=0)
            print("\n Run_model_Fixed_Fixed Results, 2 removed, 1 kept")
            print(f"For patient: {list_patients[index]} and modified_list:{modified_Patient_list[index]}, at Kept index: {index}, params are: {result}")
            TIME.sleep(240) #Adding 10 seconds, for .csv file saves
            print("--------------------------")


    if Index_remod == True:
        #Index removed, common kept
        for index in common_all_three:
            # Exclude the current index from the modified_Size1
            modified_Size1 = [x for x in Size1 if x != index]

            # The other lists remain unchanged
            modified_Inc = Inc
            modified_scaled_pop = scaled_pop
            modified_scaled_days = scaled_days
            modified_Patient_list = list_patients

            print(f"Removed index: {index}")
            # debug_function(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days, modified_Patient_list)
            result = PR_get_param_Fixed(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list,Control=0)
            print("\n Run_model_Fixed_Fixed Results, 1 removed, 2 kept")
            print(f"Removed patient: {list_patients[index]} and modified_list: For Removed index: {index}, params are:{result} ")
            TIME.sleep(240)  # Adding 60 seconds, for .csv file saves
            print("--------------------------")

    if Index_kept == True:
        #Run_model_Fixed
        for index in common_all_three:
            # Only include the current index in the modified_Size1
            modified_Size1 = [x for x in Size1 if x == index]

            # The other lists remain unchanged
            modified_Inc = Inc
            modified_scaled_pop = scaled_pop
            modified_scaled_days = scaled_days
            modified_Patient_list = list_patients

            print(f"Kept index: {index}")
            # debug_function(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list)

            # Assuming you have the actual PR_get_param function, replace the line below
            result = PR_get_param(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list)
            print("\n Run_model_Fixed Results, 2 removed, 1 kept")
            print(f"For patient: {list_patients[index]} and modified_list:{modified_Patient_list[index]}, at Kept index: {index}, params are: {result}")
            TIME.sleep(240) #Adding 10 seconds, for .csv file saves
            print("--------------------------")

    if Index_remod == True:
        #Index removed, common kept
        for index in common_all_three:
            # Exclude the current index from the modified_Size1
            modified_Size1 = [x for x in Size1 if x != index]

            # The other lists remain unchanged
            modified_Inc = Inc
            modified_scaled_pop = scaled_pop
            modified_scaled_days = scaled_days
            modified_Patient_list = list_patients

            print(f"Removed index: {index}")
            # debug_function(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days, modified_Patient_list)
            result = PR_get_param(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days,modified_Patient_list)
            print("\n Run_model_Fixed Results, 1 removed, 2 kept")
            print(f"Removed patient: {list_patients[index]} and modified_list: For Removed index: {index},, params are:{result} ")
            TIME.sleep(240)  # Adding 60 seconds, for .csv file saves
            print("--------------------------")

# # For combinations: removing one and keeping two, basicaly the combo is already done
# for combo in combinations(common_indexes, 2):
#     kept_indexes = set(combo)
#     removed_index = common_indexes - kept_indexes
#
#     modified_Size1 = [x for x in Size1 if x not in removed_index]
#     modified_Inc = [x for x in Inc if x not in removed_index]
#     modified_scaled_pop = [x for i, x in enumerate(scaled_pop) if i not in removed_index]
#     modified_scaled_days = [x for i, x in enumerate(scaled_days) if i not in removed_index]
#
#     print(f"Kept indexes: {kept_indexes}, Removed indexes: {removed_index}")
#     debug_function(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days)
#     result = PR_get_param(modified_Size1, modified_Inc, modified_scaled_pop, modified_scaled_days)
#     print(f"For Removed index: {removed_index}, Kept indexes: {kept_indexes}, params are:{result} ")
#     print("--------------------------")
#
#
#
#
#
#
#
#
# #################################OLD BUT WORKS #######################################################
#
#
# # Call the get_param function and store the result
# result = PR_get_param(Size1, Inc, scaled_pop,scaled_days) #Modified function, added scaled_pop to it
#
# print(f"result 1, Size1 Inc: {result}")
#
#
# # Call the get_param function and store the result
# result = PR_get_param(Size1, Dec, scaled_pop,scaled_days) #Modified function, added scaled_pop to
#
# print(f"result 2, Size1 Dec: {result}")
# # Call the get_param function and store the result
# result = PR_get_param(Size1, Up, scaled_pop,scaled_days) #Modified function, added scaled_pop to
#
# print(f"result 3, Size1 Up: {result}")
#
# # Call the get_param function and store the result
# result = PR_get_param(Size1, Down, scaled_pop,scaled_days) #Modified function, added scaled_pop to
#
# print(f"result 4, Size1 Down: {result}")
#
# # file_path = os.path.join(target_dir, "study_size1_param_control" + '.pkl')
# # # Save the result as a pickle file
# # with open(file_path, 'wb') as f:
# #     pickle.dump(result, f)
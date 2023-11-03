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
from multiprocessing import Pool, cpu_count
import time as TIME
from itertools import starmap


#Importing the local functions
from VG_Functions_2 import *

#Attempting something
from itertools import combinations
from datetime import datetime

#Set dataset_path, on folder back where the pickle files are located
dataset_path = r"/home/bob/project_game_theory_linux/SpiderProject_KatherLab" # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "Control_20_09_2023_attempt_7_redrawing" #C, to many patients
#Initial settings
studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential']#, 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies =['1', '2', '3', '4']

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
lim = limit(scaled_pop) #C, this lim is probably wrong

#Size1, Size2, Size4, Inc, Dec = split1(lim/2, lim*2, scaled_pop)
Size1, Size2,Size3, Size4, Inc, Dec = split1(lim/20, lim/2, lim*3, scaled_pop) #C, I think these need to be saved

#Printing
print(f"lim: {lim}") #C, the same as the ipyn: 0.003316135059100506
print(f"Size1: {len(Size1)}, {Size1}")
print(f"Size2: {len(Size2)}, {Size2}")
print(f"Size3: {len(Size3)}, {Size3}")
print(f"Size4: {len(Size4)}, {Size4}")
print(f"Inc: {len(Inc)},  {Inc}")
print(f"Dec: {len(Dec)}, {Dec}")


########################################################################
#C, This part of teh code does the filtering, by filterting the inc, decrease array index
list_d=[]
list_m=[]
list_m2=[]
# for i in range(len(list_arms)):
#   if 'DOCETAXEL' in list_arms[i] or 'Docetaxel' in list_arms[i]:
#     list_d.append((i))
#   elif 'MPDL3280A' in list_arms[i] or 'MPDL3280A_1' in list_arms[i] or 'MPDL3280A_2' in list_arms[i] or 'MPDL3280A_3' in list_arms[i] or 'Cohort 1b (Non-Squamous)' in list_arms[i] or 'Cohort 2a (Squamous)' in list_arms[i] or 'Cohort 1a (Squamous)' in list_arms[i] or:
#     list_m.append((i))
#   else:
#     list_m2.append((i))

#Chemo are DOCETAXEL, Docetaxel, everything else is immon
for i in range(len(list_arms)):
  if 'DOCETAXEL' in list_arms[i] or 'Docetaxel' in list_arms[i]:
    list_d.append((i))
  else: #Everything else immuno
    list_m.append((i))




#Creating Chemo group
Inc=[]
Dec=[]
for i in range(len(scaled_pop)):
  if i in list_d: #changed this from list_m to list_d, to get Inc only coresponding to chemotherapy group
    if scaled_pop[i][0]> scaled_pop[i][1]:
      Dec.append((i))
    else:
      Inc.append((i))


#Creating Immuno group
Inc_Immuno=[]
Dec_Immuno=[]
for i in range(len(scaled_pop)):
  if i in list_m: #changed this from list_m to list_d, to get Inc only coresponding to chemotherapy group
    if scaled_pop[i][0]> scaled_pop[i][1]:
      Dec_Immuno.append((i))
    else:
      Inc_Immuno.append((i))





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


# #Debugging
# filtered1_inc = []
# print(f" list_patients: {len(list_patients)}")
#
# #Printing corresponding patient ID for Size1 and Inc
# for i in Size1:
#     if i in Inc:
#         filtered1_inc.append(list_patients[i])
# print(f"Amount patient in Size 1 Inc: {len(set(filtered1_inc))}")
# print(f"Size1 inc: {filtered1_inc}")
#
# filtered2_inc = []
# #Printing corresponding patient ID for Size2 and Inc
# for i in Size2:
#     if i in Inc:
#         filtered2_inc.append(list_patients[i])
# print(f"Amount patient in Size 2 Inc: {len(set(filtered2_inc))}")
# print(f"Size2 inc: {filtered2_inc}")
#
# filtered3_inc = []
# #Printing corresponding patient ID for Size3 and Inc
# for i in Size3:
#     if i in Inc:
#         filtered3_inc.append(list_patients[i])
# print(f"Amount patient in Size 3 Inc: {len(set(filtered3_inc))}")
# print(f"Size3 inc: {filtered3_inc}")
#
# filtered4_inc = []
# #Printing corresponding patient ID for Size4 and Inc
# for i in Size4:
#     if i in Inc:
#         filtered4_inc.append(list_patients[i])
# print(f"Amount patient in Size 4 Inc: {len(set(filtered4_inc))}")
# print(f"Size4 inc: {filtered4_inc}")

## DETERMERNING COMMON ##

# Convert the lists to sets
Size1_set = set(Size1)
Size2_set = set(Size2)
Size3_set = set(Size3)
Size4_set = set(Size4)
Inc_set = set(Inc)
Dec_set = set(Dec)
Inc_set_immuno = set(Inc_Immuno)
Dec_set_immuno = set(Dec_Immuno)
list_d_set = set(list_d)

# Find the common numbers
common_Size1_Inc = Size1_set.intersection(Inc_set)
common_Size1_list_d = Size1_set.intersection(list_d_set)
common_Inc_list_d = Inc_set.intersection(list_d_set)
common_all_three = Size1_set.intersection(Inc_set, list_d_set)

##Common for other sizes
Size1_common = Size1_set.intersection(Inc_set,list_d_set)
Size2_common = Size2_set.intersection(Inc_set,list_d_set)
Size3_common = Size3_set.intersection(Inc_set,list_d_set) #Techinically, list_d intersection is not needed, but still bit nice to have
Size4_common = Size4_set.intersection(Inc_set,list_d_set)

##Common Dec Chemo
Size1_common_dec = Size1_set.intersection(Dec_set,list_d_set)
Size2_common_dec = Size2_set.intersection(Dec_set,list_d_set)
Size3_common_dec = Size3_set.intersection(Dec_set,list_d_set) #Techinically, list_d intersection is not needed, but still bit nice to have
Size4_common_dec = Size4_set.intersection(Dec_set,list_d_set)

###Immuno
Immuno_Size1_Inc = Size1_set.intersection(Inc_set_immuno)
Immuno_Size2_Inc = Size2_set.intersection(Inc_set_immuno)
Immuno_Size3_Inc = Size3_set.intersection(Inc_set_immuno)
Immuno_Size4_Inc = Size4_set.intersection(Inc_set_immuno)

###Immuno Dec
Immuno_Size1_Dec = Size1_set.intersection(Dec_set_immuno)
Immuno_Size2_Dec = Size2_set.intersection(Dec_set_immuno)
Immuno_Size3_Dec = Size3_set.intersection(Dec_set_immuno)
Immuno_Size4_Dec = Size4_set.intersection(Dec_set_immuno)



#########################DEBUGGING SCRIPT ###################################

##Cheking distrubtion:
# A helper function to print patient IDs for a given common variable
def print_common_patient_ids(common_var, label):
    patient_ids = [list_patients[i] for i in common_var]
    print(f"Amount of patients in {label}: {len(set(patient_ids))}")
    print(f"{label}: {patient_ids}")
    print("------")

# Printing patient IDs for each common variable
print_common_patient_ids(Size1_common, "Size1_Common_Inc")
print_common_patient_ids(Size2_common, "Size2_Common_Inc")
print_common_patient_ids(Size3_common, "Size3_Common_Inc")
print_common_patient_ids(Size4_common, "Size4_Common_Inc")

print_common_patient_ids(Size1_common_dec, "Size1_Common_Dec")
print_common_patient_ids(Size2_common_dec, "Size2_Common_Dec")
print_common_patient_ids(Size3_common_dec, "Size3_Common_Dec")
print_common_patient_ids(Size4_common_dec, "Size4_Common_Dec")

print_common_patient_ids(Immuno_Size1_Inc, "Immuno_Size1_Inc")
print_common_patient_ids(Immuno_Size2_Inc, "Immuno_Size2_Inc")
print_common_patient_ids(Immuno_Size3_Inc, "Immuno_Size3_Inc")
print_common_patient_ids(Immuno_Size4_Inc, "Immuno_Size4_Inc")

print_common_patient_ids(Immuno_Size1_Dec, "Immuno_Size1_Dec")
print_common_patient_ids(Immuno_Size2_Dec, "Immuno_Size2_Dec")
print_common_patient_ids(Immuno_Size3_Dec, "Immuno_Size3_Dec")
print_common_patient_ids(Immuno_Size4_Dec, "Immuno_Size4_Dec")


# A helper function to gather patient IDs for a given common variable
def get_common_patient_ids(common_var):
    return [list_patients[i] for i in common_var]

# Gathering patient IDs for each common variable
data = {
    "Size1_Common_Inc": get_common_patient_ids(Size1_common),
    "Size2_Common_Inc": get_common_patient_ids(Size2_common),
    "Size3_Common_Inc": get_common_patient_ids(Size3_common),
    "Size4_Common_Inc": get_common_patient_ids(Size4_common),
    "Size1_Common_Dec": get_common_patient_ids(Size1_common_dec),
    "Size2_Common_Dec": get_common_patient_ids(Size2_common_dec),
    "Size3_Common_Dec": get_common_patient_ids(Size3_common_dec),
    "Size4_Common_Dec": get_common_patient_ids(Size4_common_dec),
    "Immuno_Size1_Inc": get_common_patient_ids(Immuno_Size1_Inc),
    "Immuno_Size2_Inc": get_common_patient_ids(Immuno_Size2_Inc),
    "Immuno_Size3_Inc": get_common_patient_ids(Immuno_Size3_Inc),
    "Immuno_Size4_Inc": get_common_patient_ids(Immuno_Size4_Inc),
    "Immuno_Size1_Dec": get_common_patient_ids(Immuno_Size1_Dec),
    "Immuno_Size2_Dec": get_common_patient_ids(Immuno_Size2_Dec),
    "Immuno_Size3_Dec": get_common_patient_ids(Immuno_Size3_Dec),
    "Immuno_Size4_Dec": get_common_patient_ids(Immuno_Size4_Dec)
}

# Creating a DataFrame
df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in data.items() ]))

# Filling NaN values with an empty string (or you can choose to keep them as NaN)
df = df.fillna('')

# Saving the DataFrame to a .txt file
file_path = os.path.join(dataset_path, "patient_table_4_total_run_Fixed_grouping_study1to4.txt")
df.to_csv(file_path, index=False, sep='\t')

 #########################DEBUGGING SCRIPT ###################################

# print(f'Size 2:{Size2_common}')
# print(f'Size 3:{Size3_common}')
# print(f'Size 4:{Size4_common}')
#
# #Printing patients
# print("\n######## Size 2 ##########")
# for i in Size2_common:
#     print(f"for {i} in list_patients and in {Size2_common}: {list_patients[i]}")
# print("\n=== END OF Size 2 INFORMATION ===\n")
#
# #Printing patients
# print("\n######## Size 3 ##########")
# for i in Size3_common:
#     print(f"for {i} in list_patients and in {Size3_common}: {list_patients[i]}")
# print("\n=== END OF Size 3 INFORMATION ===\n")
#
#
# #Printing patients
# print("\n######## Size 4 ##########")
# for i in Size4_common:
#     print(f"for {i} in list_patients and in {Size4_common}: {list_patients[i]}")
# print("\n=== END OF Size 4 INFORMATION ===\n")


#C,Keeping track how long it takes to call get_param
Start_time = TIME.time()


##PARALEL CODE####
def worker_total(study_num, Size, Inc, scaled_pop, scaled_days):
    #print('It is running!!!!!')
    # Determine the type of study (Inc/Dec) from study_num
    study_type = 'Inc' if 'Inc' in study_num else 'Dec' if 'Dec' in study_num else None

    # Extract the study name e.g., 'Chemo' or 'Immuno'
    study_name = study_num.split('_')[0]

    if study_type:
        target_dir = os.path.join(dataset_path, 'Parallel_code_fixed_grouping_total_run', study_name, study_type, 'Total')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    else:
        print(f"Invalid study number format for {study_num}. Expected _Inc or _Dec in the name.")
        return

    filename = os.path.join(target_dir, f"Total_study_{study_num.rstrip(',')}_results.txt")
    with builtins.open(filename, "a", newline='') as file:
        csv_writer = csv.writer(file)

        # If it’s a fresh new file, write the header (you can adjust the header fields according to your result structure)
        if file.tell() == 0:
            csv_writer.writerow(
                ["Study Number", "mse", "k_val", "b_val", "case", "u0_val", "sigma","K", "a", "c", "g"])



        result = PR_get_param_unsave(Size, Inc, scaled_pop,scaled_days)  # replace with the actual function call

        output = (
            f"\nStudy {study_num}, "
            f"params are:{result}\n"
            "--------------------------"
        )
        #print(output)

        # Assuming the result is a tuple or list, convert it to a string or individual strings for CSV writing
        if isinstance(result, (list, tuple)):
            csv_writer.writerow([study_num,  *result])
        else:
            csv_writer.writerow([study_num,  str(result)])




def worker(study_num, common_indexes, Size, Inc, scaled_pop, scaled_days, list_patients):
    #print('It is running!!!!!')
    # Determine the type of study (Inc/Dec) from study_num
    study_type = 'Inc' if 'Inc' in study_num else 'Dec' if 'Dec' in study_num else None

    # Extract the study name e.g., 'Chemo' or 'Immuno'
    study_name = study_num.split('_')[0]

    if study_type:
        target_dir = os.path.join(dataset_path, 'Parallel_code_3_Fixed_groups_1to4', study_name, study_type, 'Index_removed')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
    else:
        print(f"Invalid study number format for {study_num}. Expected _Inc or _Dec in the name.")
        return

    filename = os.path.join(target_dir, f"study_{study_num.rstrip(',')}_results.txt")


    with builtins.open(filename, "a", newline= '') as file:
        csv_writer = csv.writer(file)

        # If it’s a fresh new file, write the header (you can adjust the header fields according to your result structure)
        if file.tell() == 0:
            csv_writer.writerow(["Study Number", "Removed Index", "Removed Patient", "mse", "k_val", "b_val", "case", "u0_val", "sigma", "K", "a", "c", "g"])


        for index in common_indexes:
            modified_Size = [x for x in Size if x != index]
            result = PR_get_param_unsave(modified_Size, Inc, scaled_pop,
                                         scaled_days)  # replace with the actual function call

            output = (
                f"\nStudy {study_num} - Removed index: {index}, "
                f"Removed patient: {list_patients[index]}, params are:{result}\n"
                "--------------------------"
            )
            #print(output)

            # Assuming the result is a tuple or list, convert it to a string or individual strings for CSV writing
            if isinstance(result, (list, tuple)):
                csv_writer.writerow([study_num, index, list_patients[index], *result])
            else:
                csv_writer.writerow([study_num, index, list_patients[index], str(result)])




studies_data = [
        ('Chemo_Size1_Inc,', Size1, Inc, scaled_pop, scaled_days),
        ('Chemo_Size2_Inc,', Size2, Inc, scaled_pop, scaled_days),
        ('Chemo_Size3_Inc', Size3, Inc, scaled_pop, scaled_days),
        ('Chemo_Size4_Inc',  Size4, Inc, scaled_pop, scaled_days),
        ('Chemo_Size1_Inc,', Size1, Inc, scaled_pop, scaled_days), #C, this one is included to determine if temp files influcences results
        ('Chemo_Size1_Dec,', Size1, Dec, scaled_pop, scaled_days),
        ('Chemo_Size2_Dec,', Size2, Dec, scaled_pop, scaled_days),
        ('Chemo_Size3_Dec', Size3, Dec, scaled_pop, scaled_days),
        ('Chemo_Size4_Dec',  Size4, Dec, scaled_pop, scaled_days),
        ('Immuno_Size1_Inc,', Size1, Inc_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size2_Inc,', Size2, Inc_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size3_Inc', Size3, Inc_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size4_Inc',  Size4, Inc_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size1_Dec,', Size1, Dec_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size2_Dec,', Size2, Dec_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size3_Dec', Size3, Dec_Immuno, scaled_pop, scaled_days),
        ('Immuno_Size4_Dec',  Size4, Dec_Immuno, scaled_pop, scaled_days)
    ]

studies_data_index = [
        ('Chemo_Size1_Inc,', Size1_common, Size1, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size2_Inc,', Size2_common, Size2, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size3_Inc', Size3_common, Size3, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size4_Inc', Size4_common, Size4, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size1_Dec,', Size1_common_dec, Size1, Dec, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size2_Dec,', Size2_common_dec, Size2, Dec, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size3_Dec', Size3_common_dec, Size3, Dec, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size4_Dec', Size4_common_dec, Size4, Dec, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size1_Inc,', Immuno_Size1_Inc, Size1, Inc_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size2_Inc,', Immuno_Size2_Inc, Size2, Inc_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size3_Inc', Immuno_Size3_Inc, Size3, Inc_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size4_Inc', Immuno_Size4_Inc, Size4, Inc_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size1_Dec,', Immuno_Size1_Dec, Size1, Dec_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size2_Dec,',  Immuno_Size2_Dec, Size2, Dec_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size3_Dec', Immuno_Size3_Dec, Size3, Dec_Immuno, scaled_pop, scaled_days, list_patients),
         ('Immuno_Size4_Dec', Immuno_Size4_Dec, Size4, Dec_Immuno, scaled_pop, scaled_days, list_patients)
 ]




#Switches
run_Total = True
run_index = False

if __name__ == '__main__':

    if run_Total == True:
        Start_time = TIME.time()
        # Prepare data for each study
        # Using multiprocessing to process each study in parallel
        with Pool(processes=cpu_count()-1) as pool:
            pool.starmap(worker_total, studies_data)

        # Pulling end TIme
        End_time = TIME.time()

        print(f"Running time: {End_time - Start_time}")

    if run_index == True:
        Start_time = TIME.time()

        # Prepare data for each study

        # Using multiprocessing to process each study in parallel
        with Pool(processes=cpu_count()-1) as pool:
            pool.starmap(worker, studies_data_index)

        # Pulling end TIme
        End_time = TIME.time()

        print(f"Running time: {End_time - Start_time}")

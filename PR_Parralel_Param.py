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


#Debugging
filtered1_inc = []
print(f" list_patients: {len(list_patients)}")

#Printing corresponding patient ID for Size1 and Inc
for i in Size1:
    if i in Inc:
        filtered1_inc.append(list_patients[i])
print(f"Amount patient in Size 1 Inc: {len(set(filtered1_inc))}")
print(f"Size1 inc: {filtered1_inc}")

filtered2_inc = []
#Printing corresponding patient ID for Size2 and Inc
for i in Size2:
    if i in Inc:
        filtered2_inc.append(list_patients[i])
print(f"Amount patient in Size 2 Inc: {len(set(filtered2_inc))}")
print(f"Size2 inc: {filtered2_inc}")

filtered3_inc = []
#Printing corresponding patient ID for Size3 and Inc
for i in Size3:
    if i in Inc:
        filtered3_inc.append(list_patients[i])
print(f"Amount patient in Size 3 Inc: {len(set(filtered3_inc))}")
print(f"Size3 inc: {filtered3_inc}")

filtered4_inc = []
#Printing corresponding patient ID for Size4 and Inc
for i in Size4:
    if i in Inc:
        filtered4_inc.append(list_patients[i])
print(f"Amount patient in Size 4 Inc: {len(set(filtered4_inc))}")
print(f"Size4 inc: {filtered4_inc}")

## DETERMERNING COMMON ##

# Convert the lists to sets
Size1_set = set(Size1)
Size2_set = set(Size2)
Size3_set = set(Size3)
Size4_set = set(Size4)
Inc_set = set(Inc)
list_d_set = set(list_d)

# Find the common numbers
common_Size1_Inc = Size1_set.intersection(Inc_set)
common_Size1_list_d = Size1_set.intersection(list_d_set)
common_Inc_list_d = Inc_set.intersection(list_d_set)
common_all_three = Size1_set.intersection(Inc_set, list_d_set)

##Common for other sizes
Size2_common = Size2_set.intersection(Inc_set,list_d_set)
Size3_common = Size3_set.intersection(Inc_set,list_d_set) #Techinically, list_d intersection is not needed, but still bit nice to have
Size4_common = Size4_set.intersection(Inc_set,list_d_set)

#Printing patients
print("\n######## Size 2 ##########")
for i in Size2_common:
    print(f"for {i} in list_patients and in {Size2_common}: {list_patients[i]}")
print("\n=== END OF Size 2 INFORMATION ===\n")

#Printing patients
print("\n######## Size 3 ##########")
for i in Size3_common:
    print(f"for {i} in list_patients and in {Size3_common}: {list_patients[i]}")
print("\n=== END OF Size 3 INFORMATION ===\n")


#Printing patients
print("\n######## Size 4 ##########")
for i in Size4_common:
    print(f"for {i} in list_patients and in {Size4_common}: {list_patients[i]}")
print("\n=== END OF Size 4 INFORMATION ===\n")


#C,Keeping track how long it takes to call get_param
Start_time = TIME.time()


##PARALEL CODE####
def worker_total(study_num, Size, Inc, scaled_pop, scaled_days, list_patients):
    target_dir = os.path.join(dataset_path, 'Prallel_code', 'Chemo', 'Inc','Total')
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filename = os.path.join(target_dir + f"study_total_{study_num}_results.txt")
    with builtins.open(filename, "a") as file:
        result = PR_get_param_unsave(Size, Inc, scaled_pop, scaled_days)  # replace with the actual function call

        output = (
            f"\nStudy {study_num} - Using all indexes, "
            f"params are:{result}\n"
            "--------------------------"
        )
        print(output)
        file.write(output)
        TIME.sleep(10)



def worker(study_num, common_indexes, Size, Inc, scaled_pop, scaled_days, list_patients):
    target_dir = os.path.join(dataset_path, 'Prallel_code','Chemo', 'Inc', 'Index_kept' )
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    filename = os.path.join(target_dir + f"study_{study_num}_results.txt")
    with builtins.open(filename, "a") as file:
        for index in common_indexes:
            modified_Size = [x for x in Size if x != index]
            result = PR_get_param_unsave(modified_Size, Inc, scaled_pop, scaled_days)  # replace with the actual function call

            output = (
                f"\nStudy {study_num} - Removed index: {index}, "
                f"Removed patient: {list_patients[index]}, params are:{result}\n"
                "--------------------------"
            )
            print(output)
            file.write(output)
            TIME.sleep(10)  # adjust the sleep time as needed

studies_data = [
        ('Chemo_Size2_Inc,', Size2, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size3_Inc', Size3, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size4_Inc',  Size4, Inc, scaled_pop, scaled_days, list_patients)
    ]

studies_data_index = [
        ('Chemo_Size2_Inc,', Size2_common, Size2, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size3_Inc', Size3_common, Size3, Inc, scaled_pop, scaled_days, list_patients),
        ('Chemo_Size4_Inc', Size4_common, Size4, Inc, scaled_pop, scaled_days, list_patients)
    ]

#Switches
run_Total = True
run_index = True

if __name__ == '__main__':

    if run_Total == True:
        Start_time = TIME.time()
        # Prepare data for each study
        # Using multiprocessing to process each study in parallel
        with Pool(processes=cpu_count()) as pool:
            pool.starmap(worker_total, studies_data)


        #Pulling end TIme
        End_time = TIME.time()

        print(f"Running time: {End_time-Start_time}")



    if run_index == True:

        # Prepare data for each study


        # Using multiprocessing to process each study in parallel
        with Pool(processes=cpu_count()) as pool:
            pool.starmap(worker, studies_data_index)

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


print(f"all studies max: {max(maxList)}")
print(f"all studies min: {min(minList)}")

##For studies 1-4
studies =['1', '2', '3', '4']

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

print(f"all studies 1 to 4: {max(maxList)}")
print(f"all studies 1 to 4: {min(minList)}")

print(maxList)


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
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.5, 0])  #
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

Size1, Size2,Size3, Size4, Inc, Dec = split1(lim/20, lim/2, lim*3, scaled_pop) #C, I think these need to be saved. The lim*2 was orginally lim*3


# Printing
print(f"lim: {lim}")

print(f"Size1: {Size1}, Size1 in Inc: {len([i for i in Size1 if i in Inc])}")
print(f"Size2: {Size2}, Size2 in Inc: {len([i for i in Size2 if i in Inc])}")
print(f"Size3: {Size3}, Size3 in Inc: {len([i for i in Size3 if i in Inc])}")
print(f"Size4: {Size4}, Size4 in Inc: {len([i for i in Size4 if i in Inc])}")

print(f"Size1: {Size1}, Size1 in Dec: {len([i for i in Size1 if i in Dec])}")
print(f"Size2: {Size2}, Size2 in Dec: {len([i for i in Size2 if i in Dec])}")
print(f"Size3: {Size3}, Size3 in Dec: {len([i for i in Size3 if i in Dec])}")
print(f"Size4: {Size4}, Size4 in Dec: {len([i for i in Size4 if i in Dec])}")

print(f"Inc: {Inc}")
print(f"Dec: {Dec}")

print(scaled_pop)


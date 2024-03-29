#C, this is next part, the previous was VG_nsclc_paper_2.py.

#C, importats from nsclc_paper.ipyn
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
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "Control_20_09_2023_attempt_8_redrawing" #C, to many patients


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
#C, now come a part of the code where the function of it is a bit vague
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


studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies =['1', '2', '3', '4', '5']
#studies = ['3']
functions=['Exponential']
###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []
first=[]
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
    #tempo = transform_to_volume(temp)
    maxList.append(max(temp)) #max value of measurement
    minList.append(min(temp))    #min value of measurement
    first.append(temp[0])

###############################################################################


print("############################################ NOW STARTING THE NEXT FOR LOOP######################################")
# Fit Funtions to the Data Points

#maxi = np.max([288, 0])
maxi = np.max(maxList)
studies = ['a', 'a', 'c', 'd', 'e']
studies =['1', '2', '3', '4', '5']
#studies=['1']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
functions =['Exponential']
splits = [True, True, False, True, True]
#splits=[False]
noPars = [3, 3, 3, 4, 3, 4]
noPars=[3,3,3,3,3,3]
limits= [0.0036254640429173885, 0.004273666526272591, 0.0024536743846284687, 0.003617425469645181, 0.0017564024423169342]
#limits = [0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824]
lim0= [2, 2, 20, 20, 2]
#lim0= [20, 20, 20, 20, 20]
#limits =[ 0.003316135059100506

r_values=[]
u_values=[]
sigma_values=[]
initial_x=[]
initial_trend=[]
target=[]

for studyName in studies:
    sind = studies.index(studyName)
    sp = splits[sind]
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True
    lim = limits[sind]
    lim= 0.003316135059100506 #depende de los grupos

    print(sind)

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

                    tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']

                    # Limit the Data Points for 6 and bigger!
                    keysList = []
                    if len(tumorFiltered_Data) >= 6:
                        dimension = list(tumorFiltered_Data['TRORRES'])
                        time = list(tumorFiltered_Data['TRDY'])

                        time = utils.Correct_Time_Vector(time, convertToWeek = True)

                        # If the value of Dimension is nan or any other string value, we replace it with zero
                        dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)

                        dimension = [x for _,x in sorted(zip(time,dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy/maxi
                            #dimension_copy = dimension_copy/np.max(dimension_copy)

                        trend = utils.Detect_Trend_Of_Data(dimension_copy)
                        #trend ='Unique'

                        dimension = [i * i * i * 0.52 for i in dimension]
                        if normalizeDimension:
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.52, 0])  #what is this line doing?
                        time.sort()
                        cn =   list(tumorFiltered_Data['TULOC']) [0]

                        try:  # C, this try box need to move to the left, so that it is in the body of the
                            Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim / 20,
                                                                                                              lim / 2,
                                                                                                              lim * 3,
                                                                                                              dimension,
                                                                                                              trend)
                            # Debug print statements
                            print(f"Size1: {Size1}, type off: {type(Size1)}, Lenght: {len(Size1)}")
                            print(f"Size2: {Size2}, type off: {type(Size2)}, Lenght:  {len(Size2)}")
                            print(f"Size3: {Size3}, type off: {type(Size3)}, Lenght:  {len(Size3)}")
                            print(f"Size4: {Size4}, type off: {type(Size4)}, Lenght:  {len(Size4)}")
                            print(f"Inc: {Inc}, type off: {type(Inc)}, Lenght: {len(Inc)}")
                            print(f"Dec: {Dec}, type off: {type(Dec)}, Lenght:  {len(Dec)}")

                            # #C, currently their are sometimes empty array for that patient, and this make the code crash
                            # # Check and insert 2 into each empty Size array
                            # if len(Size1) == 0:
                            #     Size1.append(2)
                            # if len(Size2) == 0:
                            #     Size2.append(2)
                            # if len(Size3) == 0:
                            #     Size3.append(2)
                            # if len(Size4) == 0:
                            #     Size4.append(2)

                            # Debug print statements
                            print(f"Size1: {Size1}, type off: {type(Size1)}, Lenght: {len(Size1)}")
                            print(f"Size2: {Size2}, type off: {type(Size2)}, Lenght:  {len(Size2)}")
                            print(f"Size3: {Size3}, type off: {type(Size3)}, Lenght:  {len(Size3)}")
                            print(f"Size4: {Size4}, type off: {type(Size4)}, Lenght:  {len(Size4)}")
                            print(f"Inc: {Inc}, type off: {type(Inc)}, Lenght: {len(Inc)}")
                            print(f"Dec: {Dec}, type off: {type(Dec)}, Lenght:  {len(Dec)}")
                            print(f"dimension: {dimension}")

                            Fluctuate.extend(Evolution)
                            # C, commented the below line trying to do something new to fix it
                            # k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName, dimension, arm)

                            # C, made a new seprate by size to try to work around empty list size
                            k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size(studyName, arm,
                                                                                                   Size1, Size2, Size3,
                                                                                                   Size4, Up, Down,
                                                                                                   Fluctuate, Evolution,
                                                                                                   Inc, Dec)
                            print(f"separate_by_size is runned")
                            list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=time,
                                                                                                    population=dimension,
                                                                                                    case=case0,
                                                                                                    k_val=k0,
                                                                                                    b_val=b0, u0_val=u0,
                                                                                                    sigma_val=sigma0,
                                                                                                    Kmax0=K0, a_val=a0,
                                                                                                    c_val=c0,
                                                                                                    free='sigma',
                                                                                                    g_val=g0)

                            initial_x.append(dimension[0])
                            if len(Inc) == 1:
                                initial_trend.append(1)
                            else:
                                initial_trend.append(0)

                            r_values.append(list_Kmax)
                            u_values.append(list_u[0])
                            sigma_values.append(list_s)

                            if list_x[-1] > list_x[0]:
                                target.append(0)
                            else:
                                target.append(1)

                            modelPredictions = list_x

                            if trend == 'Evolution':
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                                fig.suptitle("Exponential cost of resistance " + arm, fontsize=15)

                                # Change scatter plot to gray line for ax1
                                ax1.plot(time, list_u, label='u', color='gray')
                                ax1.legend(fontsize=15)

                                # For ax2, use scatter for red crosses and plot for blue line
                                ax2.scatter(time, dimension, label="real measurements", color='red',
                                            marker='x')  # red crosses
                                ax2.plot(time, list_x, label='model predictions', color='blue')  # blue line
                                ax2.legend(fontsize=15)

                                ax1.set_xlabel("days from treatment start", fontsize=15)
                                ax1.set_ylabel("value of u", fontsize=15)
                                ax2.set_xlabel("days from treatment start", fontsize=15)
                                ax2.set_ylabel("volume of tumor", fontsize=15)

                                # Create the folder if it doesn't exist.
                                save_path = os.path.join(dataset_path, "Sigma 0", functionToFit,
                                                         f"{arm}")  # added arm to it
                                if not os.path.exists(save_path):
                                    os.makedirs(save_path)

                                # Save the figure
                                fig.savefig(os.path.join(save_path, str(key)))



                                ##C, Old plot function
                                # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                                # fig.suptitle("Exponential cost of resistance " + arm, fontsize=12)
                                # ax1.scatter(time, list_u, label='u', color='black', linestyle='dashed')
                                # ax1.legend(fontsize=12)
                                #
                                # # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                                # # ax2.set_title("m: " + str(list_Kmax1))
                                # ax2.plot(time, dimension, label="real measurements", color='red')
                                # ax2.plot(time, list_x, label='model predictions', color='blue')
                                # ax2.legend(fontsize=12)
                                # ax1.set_xlabel("days from treatment start", fontsize=12)
                                # ax1.set_ylabel("value of u", fontsize=12)
                                # ax2.set_xlabel("days from treatment start", fontsize=12)
                                # ax2.set_ylabel("volume of tumor", fontsize=12)
                                # # Create the folder if it doesn't exist.
                                # save_path = os.path.join(dataset_path, "Sigma 0", functionToFit, f"{arm}") #added arm to it
                                # if not os.path.exists(save_path):
                                #     os.makedirs(save_path)
                                #
                                # # Save the figure
                                # fig.savefig(os.path.join(save_path, str(key)))
                                # # C, modifyed this line, therefore this is commented
                                # # fig.savefig(dataset_path + "Sigma 0/" + functionToFit + str(key))

                            if len(set(dimension)) == 1:
                                modelPredictions = dimension
                            else:
                                modelPredictions = list_x

                            modelPredictions = [0 if str(i) == 'nan' else i for i in modelPredictions]

                            absError = modelPredictions - dimension
                            SE = np.square(absError)
                            temp_sum = np.sum(SE)
                            MSE = np.mean(SE)

                            # Saving Params

                            # Ensure the directory exists, if not, create it
                            directory_parm = os.path.join(dataset_path, functionToFit, paramControl,
                                                          f"{studyName}")  # C, I have to many patients so let me determine which one is used
                            if not os.path.exists(directory_parm):
                                os.makedirs(directory_parm)

                            # Create the CSV file path for this patient
                            # Create the CSV file path for this patient
                            csv_file_path = os.path.join(directory_parm, f"{key}.csv")

                            # Find the max length among all variables to extend them
                            max_len = 0
                            # Calculate max length to extend arrays to this size
                            all_variables = [key, Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec,
                                             lim,
                                             list_Kmax, list_u[0], list_s, absError, SE, temp_sum, MSE]

                            for var in all_variables:
                                try:
                                    if isinstance(var, (list, np.ndarray)):
                                        max_len = max(max_len, len(var))
                                    elif isinstance(var, (int, float, np.float64, str)):
                                        max_len = max(max_len, 1)
                                    else:
                                        print(f"Error: Unrecognized type {type(var)} for variable {var}")
                                except Exception as e:
                                    print(f"Exception while calculating length for variable {var}: {e}")

                            # Extend all arrays to max_length
                            extended_variables = []
                            for var in all_variables:
                                try:
                                    extended_var = extend_to_length(var, max_len)
                                    extended_variables.append(extended_var)
                                except Exception as e:
                                    print(f"Exception while extending variable {var}: {e}")
                                    extended_variables.append([var] * max_len)  # Default action

                            # Write to the CSV file
                            with open(csv_file_path, 'w', newline='') as csvfile:
                                csvwriter = csv.writer(csvfile)

                                # Writing the header
                                headers = ["PatientID", "Size1", "Size2", "Size3", "Size4", "Up", "Down", "Fluctuate",
                                           "Evolution",
                                           "Inc", "Dec", "lim", "list_Kmax(r_values)", "list_u[0](u_values)",
                                           "list_s(sigma values)", "absError", "SE", "temp_sum", "MSE"]
                                csvwriter.writerow(headers)

                                # Writing the data
                                for i in range(max_len):
                                    row = [extended_variables[j][i] for j in range(len(extended_variables))]
                                    csvwriter.writerow(row)

                            result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
                                                                     categories=['patientID', 'time', 'dimension',
                                                                                 'prediction', 'rmse', 'rSquare', 'aic',
                                                                                 'params', 'cancer'],
                                                                     values=[key, time, dimension, modelPredictions,
                                                                             mean_squared_error(dimension,
                                                                                                modelPredictions),
                                                                             r2_score(dimension, modelPredictions),
                                                                             (2 * noParameters) - (
                                                                                         2 * np.log(temp_sum)),
                                                                             group, cn])

                        except Exception as e:
                            print(f"An error occurred for patient {key} and arm {arm}: {e}")
                            continue

                    # try: #C, this try box need to move to the left, so that it is in the body of the
                    #     Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim / 20,
                    #                                                                                       lim / 2,
                    #                                                                                       lim * 3,
                    #                                                                                       dimension,
                    #                                                                                       trend)
                    #     # Debug print statements
                    #     print(f"Size1: {Size1}, type off: {type(Size1)}, Lenght: {len(Size1)}")
                    #     print(f"Size2: {Size2}, type off: {type(Size2)}, Lenght:  {len(Size2)}")
                    #     print(f"Size3: {Size3}, type off: {type(Size3)}, Lenght:  {len(Size3)}")
                    #     print(f"Size4: {Size4}, type off: {type(Size4)}, Lenght:  {len(Size4)}")
                    #     print(f"Inc: {Inc}, type off: {type(Inc)}, Lenght: {len(Inc)}")
                    #     print(f"Dec: {Dec}, type off: {type(Dec)}, Lenght:  {len(Dec)}")
                    #
                    #     # #C, currently their are sometimes empty array for that patient, and this make the code crash
                    #     # # Check and insert 2 into each empty Size array
                    #     # if len(Size1) == 0:
                    #     #     Size1.append(2)
                    #     # if len(Size2) == 0:
                    #     #     Size2.append(2)
                    #     # if len(Size3) == 0:
                    #     #     Size3.append(2)
                    #     # if len(Size4) == 0:
                    #     #     Size4.append(2)
                    #
                    #
                    #     # Debug print statements
                    #     print(f"Size1: {Size1}, type off: {type(Size1)}, Lenght: {len(Size1)}")
                    #     print(f"Size2: {Size2}, type off: {type(Size2)}, Lenght:  {len(Size2)}")
                    #     print(f"Size3: {Size3}, type off: {type(Size3)}, Lenght:  {len(Size3)}")
                    #     print(f"Size4: {Size4}, type off: {type(Size4)}, Lenght:  {len(Size4)}")
                    #     print(f"Inc: {Inc}, type off: {type(Inc)}, Lenght: {len(Inc)}")
                    #     print(f"Dec: {Dec}, type off: {type(Dec)}, Lenght:  {len(Dec)}")
                    #     print(f"dimension: {dimension}")
                    #
                    #     Fluctuate.extend(Evolution)
                    #     #C, commented the below line trying to do something new to fix it
                    #     #k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName, dimension, arm)
                    #
                    #     #C, made a new seprate by size to try to work around empty list size
                    #     k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size(studyName, arm, Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
                    #     print(f"separate_by_size is runned")
                    #     list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=time,
                    #                                                                             population=dimension,
                    #                                                                             case=case0, k_val=k0,
                    #                                                                             b_val=b0, u0_val=u0,
                    #                                                                             sigma_val=sigma0,
                    #                                                                             Kmax0=K0, a_val=a0,
                    #                                                                             c_val=c0, free='sigma',
                    #                                                                             g_val=g0)
                    #
                    #     initial_x.append(dimension[0])
                    #     if len(Inc) == 1:
                    #         initial_trend.append(1)
                    #     else:
                    #         initial_trend.append(0)
                    #
                    #     r_values.append(list_Kmax)
                    #     u_values.append(list_u[0])
                    #     sigma_values.append(list_s)
                    #
                    #     if list_x[-1] > list_x[0]:
                    #         target.append(0)
                    #     else:
                    #         target.append(1)
                    #
                    #     modelPredictions = list_x
                    #
                    #     if trend == 'Evolution':
                    #         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                    #         fig.suptitle("Exponential cost of resistance " + arm, fontsize=12)
                    #         ax1.scatter(time, list_u, label='u', color='black', linestyle='dashed')
                    #         ax1.legend(fontsize=12)
                    #
                    #         # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                    #         # ax2.set_title("m: " + str(list_Kmax1))
                    #         ax2.plot(time, dimension, label="real measurements", color='red')
                    #         ax2.plot(time, list_x, label='model predictions', color='blue')
                    #         ax2.legend(fontsize=12)
                    #         ax1.set_xlabel("days from treatment start", fontsize=12)
                    #         ax1.set_ylabel("value of u", fontsize=12)
                    #         ax2.set_xlabel("days from treatment start", fontsize=12)
                    #         ax2.set_ylabel("volume of tumor", fontsize=12)
                    #         # Create the folder if it doesn't exist.
                    #         save_path = os.path.join(dataset_path, "Sigma 0", functionToFit)
                    #         if not os.path.exists(save_path):
                    #             os.makedirs(save_path)
                    #
                    #         # Save the figure
                    #         fig.savefig(os.path.join(save_path, str(key)))
                    #         #C, modifyed this line, therefore this is commented
                    #         #fig.savefig(dataset_path + "Sigma 0/" + functionToFit + str(key))
                    #
                    #     if len(set(dimension)) == 1:
                    #         modelPredictions = dimension
                    #     else:
                    #         modelPredictions = list_x
                    #
                    #     modelPredictions = [0 if str(i) == 'nan' else i for i in modelPredictions]
                    #
                    #     absError = modelPredictions - dimension
                    #     SE = np.square(absError)
                    #     temp_sum = np.sum(SE)
                    #     MSE = np.mean(SE)
                    #
                    #     #Saving Params
                    #
                    #     # Ensure the directory exists, if not, create it
                    #     directory_parm = os.path.join(dataset_path, functionToFit, paramControl, f"{studyName}") #C, I have to many patients so let me determine which one is used
                    #     if not os.path.exists(directory_parm):
                    #         os.makedirs(directory_parm)
                    #
                    #     # Create the CSV file path for this patient
                    #     # Create the CSV file path for this patient
                    #     csv_file_path = os.path.join(directory_parm, f"{key}.csv")
                    #
                    #     # Find the max length among all variables to extend them
                    #     max_len = 0
                    #     # Calculate max length to extend arrays to this size
                    #     all_variables = [key, Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec, lim,
                    #                      list_Kmax, list_u[0], list_s, absError, SE, temp_sum, MSE]
                    #
                    #     for var in all_variables:
                    #         try:
                    #             if isinstance(var, (list,np.ndarray)):
                    #                 max_len = max(max_len, len(var))
                    #             elif isinstance(var, (int, float, np.float64, str)):
                    #                 max_len = max(max_len, 1)
                    #             else:
                    #                 print(f"Error: Unrecognized type {type(var)} for variable {var}")
                    #         except Exception as e:
                    #             print(f"Exception while calculating length for variable {var}: {e}")
                    #
                    #     # Extend all arrays to max_length
                    #     extended_variables = []
                    #     for var in all_variables:
                    #         try:
                    #             extended_var = extend_to_length(var, max_len)
                    #             extended_variables.append(extended_var)
                    #         except Exception as e:
                    #             print(f"Exception while extending variable {var}: {e}")
                    #             extended_variables.append([var] * max_len)  # Default action
                    #
                    #     # Write to the CSV file
                    #     with open(csv_file_path, 'w', newline='') as csvfile:
                    #         csvwriter = csv.writer(csvfile)
                    #
                    #         # Writing the header
                    #         headers = ["PatientID","Size1", "Size2", "Size3", "Size4", "Up", "Down", "Fluctuate", "Evolution",
                    #                    "Inc", "Dec", "lim", "list_Kmax(r_values)", "list_u[0](u_values)",
                    #                    "list_s(sigma values)", "absError", "SE", "temp_sum", "MSE"]
                    #         csvwriter.writerow(headers)
                    #
                    #         # Writing the data
                    #         for i in range(max_len):
                    #             row = [extended_variables[j][i] for j in range(len(extended_variables))]
                    #             csvwriter.writerow(row)
                    #
                    #     result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
                    #                                              categories=['patientID', 'time', 'dimension',
                    #                                                          'prediction', 'rmse', 'rSquare', 'aic',
                    #                                                          'params', 'cancer'],
                    #                                              values=[key, time, dimension, modelPredictions,
                    #                                                      mean_squared_error(dimension,
                    #                                                                         modelPredictions),
                    #                                                      r2_score(dimension, modelPredictions),
                    #                                                      (2 * noParameters) - (2 * np.log(temp_sum)),
                    #                                                      group, cn])
                    #
                    # except Exception as e:
                    #     print(f"An error occurred for patient {key} and arm {arm}: {e}")
                    #     continue
        #Debugging,
        print(f"functionToFit: {functionToFit}")
        print(f"result_dict: {result_dict}")
        print(f"list: {list_b}")


        # Properly join paths and check if directory exists
        full_directory_path = os.path.join(dataset_path, functionToFit)

        # Check if directory exists and if not, create it
        if not os.path.exists(full_directory_path):
            os.makedirs(full_directory_path)

        # Now join the path for the file
        file_path = os.path.join(full_directory_path, studyName + '.pkl')

        with builtins.open(file_path, "wb") as a_file:  # Using with ensures the file will be closed after the block
            pickle.dump(result_dict, a_file)

        a_file.close()







# for studyName in studies:
#     sind = studies.index(studyName)
#     sp = splits[sind]
#     studyName = studies[sind]
#     warnings.filterwarnings("ignore")
#     normalizeDimension = True
#     lim = limits[sind]
#     lim= 0.003316135059100506 #depende de los grupos
#
#     print(sind)
#
#     #rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
#     rawDataPath = os.path.join(dataset_path,  'Study_' + studyName + '_1.xlsx')
#     data, arms = utils.Read_Excel(rawDataPath, ArmName = 'TRT01A', split = sp)
#     for functionToFit in functions:
#
#         find = functions.index(functionToFit)
#         noParameters = noPars[find]
#         result_dict = utils.Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate', 'Evolution'], categories = ['patientID', 'rmse', 'rSquare',
#                                                                                                 'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
#         print(functionToFit)
#         print(studyName)
#         for arm in arms:
#             print(arm)
#             data_temp = data.loc[data['receivedTreatment'] == arm]
#             patientID = list(data_temp['USUBJID'].unique())
#
#             for key in patientID:
#
#                 filteredData = data.loc[data['USUBJID'] == key]
#                 temp = filteredData['TRLINKID'].unique()
#                 temp = [i for i in temp if not str(i) == 'nan']
#                 temp = [i for i in temp if not '-NT' in str(i)]
#
#                 if  'INV-T001' in temp :
#                     tumorFiltered_Data = filteredData.loc[filteredData['TRLINKID'] == 'INV-T001']
#                     tumorFiltered_Data.dropna(subset = ['TRDY'], inplace = True)
#
#                     tumorFiltered_Data = tumorFiltered_Data.loc[tumorFiltered_Data['TRTESTCD'] == 'LDIAM']
#
#                     # Limit the Data Points for 6 and bigger!
#                     keysList = []
#                     if len(tumorFiltered_Data) >= 6:
#                         dimension = list(tumorFiltered_Data['TRORRES'])
#                         time = list(tumorFiltered_Data['TRDY'])
#
#                         time = utils.Correct_Time_Vector(time, convertToWeek = True)
#
#                         # If the value of Dimension is nan or any other string value, we replace it with zero
#                         dimension = utils.Remove_String_From_Numeric_Vector(dimension, valueToReplace = 0)
#
#                         dimension = [x for _,x in sorted(zip(time,dimension))]
#                         dimension_copy = dimension.copy()
#                         if normalizeDimension:
#                             dimension_copy = dimension_copy/maxi
#                             #dimension_copy = dimension_copy/np.max(dimension_copy)
#
#                         trend = utils.Detect_Trend_Of_Data(dimension_copy)
#                         #trend ='Unique'
#
#                         dimension = [i * i * i * 0.52 for i in dimension]
#                         if normalizeDimension:
#                             dimension = dimension/np.max([maxi * maxi * maxi * 0.52, 0])  #what is this line doing?
#                         time.sort()
#                         cn =   list(tumorFiltered_Data['TULOC']) [0]
#
#                         try:
#                         #if True:
#                           Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/20, lim/2, lim*3, dimension, trend)
#                         # Debug print statements
#                             print(f"Size1: {Size1}")
#                             print(f"Size2: {Size2}")
#                             print(f"Size3: {Size3}")
#                             print(f"Size4: {Size4}")
#                             print(f"Inc: {Inc}")
#                             print(f"Dec: {Dec}")
#                           #Size1, Size2,Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/lim0[sind], lim/2, lim*3, dimension, trend) #es lim*3
#                           '''if sind ==0 or sind ==1 or sind ==4:
#                             Size2= Size3 '''
#
#
#                           Fluctuate.extend(Evolution) ##comment if 4 groupd
#                           k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size( studyName, dimension, arm)
#                           #k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m(dimension)
#
#
#
#                           list_x, list_u, list_Kmax, error, list_b, list_s, der =run_model_fixed(days=time, population= dimension,case=case0, k_val=k0, b_val=b0, u0_val=u0, sigma_val=sigma0, Kmax0=K0, a_val=a0, c_val=c0, free='sigma', g_val=g0)
#
#                           initial_x.append(dimension[0])
#                           if len(Inc)==1:
#                             initial_trend.append(1)
#                           else:
#                             initial_trend.append(0)
#                           r_values.append(list_Kmax)
#                           u_values.append(list_u[0])
#                           sigma_values.append(list_s)
#                           if list_x[-1] > list_x[0]:
#                             target.append(0)
#                           else:
#                             target.append(1) #1 means success
#                           modelPredictions = list_x
#                           #print(dimension- list_x)
#                           #print('pred: ' + str(list_x))
#                           if trend == 'Evolution':
#                             fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
#                             fig.suptitle("Exponential cost of resistance "  + arm, fontsize=12)
#                             ax1.scatter(time, list_u, label='u', color='black', linestyle='dashed')
#                             ax1.legend( fontsize=12)
#
#                             #ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
#                             #ax2.set_title("m: " + str(list_Kmax1))
#                             ax2.plot(time, dimension, label="real measurements", color='red')
#                             ax2.plot(time, list_x, label='model predictions', color='blue')
#                             ax2.legend( fontsize=12)
#                             ax1.set_xlabel("days from treatment start", fontsize=12)
#                             ax1.set_ylabel("value of u", fontsize=12)
#                             ax2.set_xlabel("days from treatment start", fontsize=12)
#                             ax2.set_ylabel("volume of tumor", fontsize=12)
#                             fig.savefig(dataset_path + "Sigma 0/" +str( key))
#
#                           if len(set(dimension)) == 1:
#                               modelPredictions = dimension
#                           else:
#                               modelPredictions =  list_x
#
#                           modelPredictions = [0 if str(i) == 'nan' else i  for i in modelPredictions]
#                           absError = modelPredictions - dimension
#                           SE = np.square(absError)
#                           temp_sum = np.sum(SE)
#                           MSE = np.mean(SE)
#
#
#                           result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'],
#                                                                             values = [key, time, dimension, modelPredictions, mean_squared_error(dimension, modelPredictions),
#                                                                                       r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), group, cn]) #need to put parameter
#                         except Exception as e:
#                             print(f"An error occurred for patient {key} and arm {arm}: {e}")
#                             continue
#
#         #Debugging,
#         print(f"functionToFit: {functionToFit}")
#         print(f"result_dict: {result_dict}")
#
#
#         # Properly join paths and check if directory exists
#         full_directory_path = os.path.join(dataset_path, functionToFit)
#
#         # Check if directory exists and if not, create it
#         if not os.path.exists(full_directory_path):
#             os.makedirs(full_directory_path)
#
#         # Now join the path for the file
#         file_path = os.path.join(full_directory_path, studyName + '.pkl')
#
#         with builtins.open(file_path, "wb") as a_file:  # Using with ensures the file will be closed after the block
#             pickle.dump(result_dict, a_file)
#
#         a_file.close()


print("######################################################FOR LOOP CLOSED #####################################################")
print(f"result.dict: {result_dict}")

result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
                                         categories=['patientID', 'time', 'dimension', 'prediction', 'rmse', 'rSquare',
                                                     'aic', 'params', 'cancer'],
                                         values=[key, time, dimension, modelPredictions,
                                                 mean_squared_error(dimension, modelPredictions),
                                                 r2_score(dimension, modelPredictions),
                                                 (2 * noParameters) - (2 * np.log(temp_sum)), group, cn])

print(f"result.dict: {result_dict}")


#Libraries
import os

import Utils as utils
from VG_Functions import *
import numpy as np
import pickle
from os import *
from sklearn.metrics import r2_score
import FitFunctions as ff
import warnings
from scipy.optimize import curve_fit
import sys
import json

#C, checking if this make open work
import builtins

#dataset_path = folder_path = "C:\\Users\\Shade\\Desktop\\Master\\Project Game Theory Code\\Downloaded file\\Edited\\SpiderProject_KatherLab"
dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)

# copy dataset into working directory
#dataset_path = 'SpiderProject_KatherLab'


##NARMINS CODE#####

###############################################################################

studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies =['1', '2', '3', '4', '5']
#studies=['1']
#functions=['Exponential']

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
    maxList.append(max(temp)) #max value of measurement
    minList.append(min(temp))    #min value of measurement
###############################################################################

# Fit Funtions to the Data Points

#maxi = np.max([288, 0])
maxi = np.max(maxList)

studies = ['a', 'a', 'c', 'd', 'e']
studies =['1', '2', '3', '4', '5']

functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
functions =['GeneralBertalanffy']
splits = [True, True, False, True, True]
noPars = [3, 3, 3, 4, 3, 4]
noPars = [4, 3, 3, 4, 3, 4]

#studies=['1']
#functions=['Exponential']


for studyName in studies:
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

                                                                                                #'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
        #result_dict = utils.Create_Result_dict(arms, ['Unique'], categories = ['patientID', 'rmse', 'rSquare',
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
                        #trend = 'Unique'


                        dimension = [i * i * i * 0.52 for i in dimension] #?
                        if normalizeDimension:
                            dimension = dimension/np.max([maxi * maxi * maxi * 0.52, 0]) #what is this line doing
                        time.sort()
                        cn =   list(tumorFiltered_Data['TULOC']) [0]
                        param_bounds=([0] *noParameters ,[np.inf] * noParameters)
                        if find == 4:
                            param_bounds=([[0, -np.inf, 0],[np.inf, np.inf, np.inf]])
                        elif find == 5:
                            param_bounds=([[0, -np.inf, 2/3, 0],[np.inf, np.inf, 1, np.inf]])
                        elif find == 3:
                            param_bounds=([[0, 0, 2/3, 0],[np.inf, np.inf, 1, np.inf]])
                        #print(dimension)
                        #firstDim = dimension[0:-3]
                        #firstTime = time[0:-3]

                        try:
                            fitfunc = ff.Select_Fucntion(functionToFit)
                            geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = time, d = dimension)
                            fittedParameters, pcov = curve_fit(fitfunc, time, dimension, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf')
                            modelPredictions = fitfunc(time, *fittedParameters)    # this is the list of values predicted with model
                        except:
                            try:
                                geneticParameters = ff.generate_Initial_Parameters_genetic(fitfunc,k = noParameters, boundry = [0, 1], t = time, d = dimension)
                                if len(param_bounds[0]) == 4:
                                   geneticParameters[0] = 0.001
                                   geneticParameters[1] = 0.001
                                   geneticParameters[2] = 0.7
                                else:
                                   geneticParameters[0] = 0.001
                                   geneticParameters[1] = 0.7
                                fittedParameters, pcov = curve_fit(fitfunc, time, dimension, geneticParameters, maxfev = 200000, bounds = param_bounds, method = 'trf')
                                modelPredictions = fitfunc(time, *fittedParameters)
                            except:
                                print(key)
                                result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'],
                                                                          values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn])
                                continue

                        if len(set(dimension)) == 1:
                            modelPredictions = dimension
                        else:
                            modelPredictions = fitfunc(time, *fittedParameters)

                        modelPredictions = [0 if str(i) == 'nan' else i  for i in modelPredictions]
                        absError = modelPredictions - dimension
                        SE = np.square(absError)
                        temp_sum = np.sum(SE)
                        MSE = np.mean(SE)

                        result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'],
                                                                          values = [key, time, dimension, modelPredictions, mean_squared_error(dimension, modelPredictions),
                                                                                    r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), fittedParameters, cn])
                        print(f"result dict: {result_dict}")
        #Debugging
        print(type(dataset_path))
        print(type(functionToFit))
        print(type(studyName))
        print("dataset_path:", dataset_path, type(dataset_path))
        print("functionToFit:", functionToFit, type(functionToFit))
        print("studyName:", studyName, type(studyName))

        #a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        #a_file = open(os.path.join(dataset_path, functionToFit, {studyName} + '.pkl'), "wb") #C, this line doesnt work for me
        #a_file = open(os.path.join(dataset_path, functionToFit, studyName + ".pkl"), "wb")


        #Pickle file saving

        # Inside your loop
        output_directory = os.path.join(dataset_path, functionToFit)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)  # Create directory if it doesn't exist

        output_file_path = os.path.join(output_directory, f"{studyName}.pkl")
        #Debugging
        print(f"Output directory: {output_directory}, type: {type(output_directory)}")
        print(f"Study name: {studyName}, type: {type(studyName)}")
        print(f"Output file path: {output_file_path}, type: {type(output_file_path)}")

        with builtins.open(output_file_path, "wb") as a_file:  # Using with ensures the file will be closed after the block
            pickle.dump(result_dict, a_file)



        a_file.close()

        #JSON, attempt gives an error because int64 object is not natively serializable to JSON
        #output_file_path = os.path.join(output_directory, f"{studyName}.json")  # Changing the extension to .json
        #Debugging
        # print(f"Output directory: {output_directory}, type: {type(output_directory)}")
        # print(f"Study name: {studyName}, type: {type(studyName)}")
        # print(f"Output file path: {output_file_path}, type: {type(output_file_path)}")
        # print(f"result dict: {result_dict}")
        # with builtins.open("{output_file_path}", "w") as a_file:  # Open the file in write mode
        #     json.dump(result_dict, a_file, indent=4)  # Serialize the Python object to a JSON formatted string

    # Removed the line `a_file.close()` because the `with` statement will handle closing the file

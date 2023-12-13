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
dataset_path = r"/home/bob/project_game_theory_linux/SpiderProject_KatherLab" # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "4_November" #C, to many patients



"""#opt"""

studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
studies = ['1', '2', '3', '4', '5']
# studies = ['3']
functions = ['Exponential']
###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []
first = []
for studyName in studies:
    # rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path, 'Study_' + studyName + '_1.xlsx')
    sind = studies.index(studyName)
    sp = splits[sind]
    data, arms = utils.Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']  # take only first tumor
    filtered_Data = filtered_Data.loc[
        filtered_Data['TRTESTCD'] == 'LDIAM']  # take only tumors for which measurement of longer diameter is available
    temp = list(filtered_Data['TRORRES'])  # this should be the measurements
    temp = utils.Remove_String_From_Numeric_Vector(temp,
                                                   valueToReplace=0)  # removes strings and replace by zero, why? do we only have strings when it disappears?
    # tempo = transform_to_volume(temp)
    maxList.append(max(temp))  # max value of measurement
    minList.append(min(temp))  # min value of measurement
    first.append(temp[0])

###############################################################################

# Fit Funtions to the Data Points

# maxi = np.max([288, 0])
maxi = np.max(maxList)
studies = ['a', 'a', 'c', 'd', 'e']
studies = ['1', '2', '3', '4']
# studies=['1']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
functions = ['Exponential']
splits = [True, True, False, True, True]
# splits=[False]
noPars = [3, 3, 3, 4, 3, 4]
noPars = [3, 3, 3, 3, 3, 3]
limits = [0.0036254640429173885, 0.004273666526272591, 0.0024536743846284687, 0.003617425469645181,
          0.0017564024423169342]
# limits = [0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824, 0.0030320564222189824]
lim0 = [2, 2, 20, 20, 2]
# lim0= [20, 20, 20, 20, 20] 003448767


for studyName in studies:
    sind = studies.index(studyName)
    sp = splits[sind]
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True
    lim = limits[sind]
    lim = 0.003316135059100506
    print(sind)

    # rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path, 'Study_' + studyName + '_1.xlsx')
    data, arms = utils.Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    for functionToFit in functions:

        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = utils.Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate', 'Evolution'],
                                               categories=['patientID', 'rmse', 'rSquare',
                                                           'time', 'dimension', 'prediction', 'aic', 'params',
                                                           'cancer'])
        # result_dict = utils.Create_Result_dict(arms, ['Unique'], categories = ['patientID', 'rmse', 'rSquare',
        # 'time', 'dimension', 'prediction', 'aic', 'params', 'cancer'])
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
                        # trend ='Unique'

                        dimension = [i * i * i * 0.52 for i in dimension]
                        if normalizeDimension:
                            dimension = dimension / np.max([maxi * maxi * maxi * 0.52, 0])  # what is this line doing?
                        time.sort()
                        cn = list(tumorFiltered_Data['TULOC'])[0]

                        # scale my way
                        # diemnsion = transform_to_volume(dimension)
                        # dimension = scale_data(dimension, maxi)
                        # scaled_days.append(time)
                        # scaled_pop.append(dimension)

                        # firstDim = dimension[0:-1]
                        # firstTime = time[0:-1]
                        firstDim = dimension
                        firstTime = time

                        try:
                            # if True:
                            #if not ('DOCETAXEL' in arm or 'Docetaxel' in arm):
                            #C, immuno off
                            #if arm == 'DOCETAXEL' or arm == 'Docetaxel': #chemo off is: if arm != 'DOCETAXEL' and arm != 'Docetaxel':
                            #C, chemo off
                            if arm == 'DOCETAXEL' or arm == 'Docetaxel':
                                print('Running')
                                # Size1, Size2, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split_ind(lim/2, lim*2, dimension, trend)
                                # Size1, Size2,Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/lim0[sind], lim/2, lim*3, dimension, trend) #es lim*3
                                Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(
                                    lim / 20, lim / 2, lim * 3, dimension, trend)
                                '''if sind ==0 or sind ==1 or sind ==4:
                                  Size2= Size3 '''

                                Fluctuate.extend(Evolution)  ##comment if 4 groupd
                                k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_chemo_vs_immuno(studyName, arm,
                                                                                                       Size1, Size2, Size3,
                                                                                                       Size4, Up, Down,
                                                                                                       Fluctuate, Evolution,
                                                                                                       Inc, Dec)
                                # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all(dimension)

                                time1 = time + [time[-1] + 10, time[-1] + 20, time[-1] + 30, time[-1] + 40,
                                                time[-1] + 50]
                                # time1=time
                                # time1 = np.linspace(time[0], time[-1], 20, endpoint=True)

                                list_x, list_u_correct, list_Kmax_correct, error, list_b_correct, list_s_correct, der = run_model_fixed_unsaved(
                                    days=time,
                                    population=dimension,
                                    case=case0,
                                    k_val=k0,
                                    b_val=b0, u0_val=u0,
                                    sigma_val=sigma0,
                                    Kmax0=K0, a_val=a0,
                                    c_val=c0,
                                    free='sigma',
                                    g_val=g0)

                                # list_x0, list_u0, list_Kmax0, error0, list_b0, list_s0, der0 = run_model_fixed_unsaved(
                                #     days=time, population=firstDim, case=case0, k_val=k0, b_val=b0, u0_val=u0,
                                #     sigma_val=sigma0, Kmax0=K0, a_val=a0, c_val=c0, free='k', g_val=g0)

                                # what would have happened to docetaxel if they had been given immuno
                                # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size( studyName, dimension, 'Docetaxel')

                                list_x, list_u, list_Kmax, error, list_b, list_s = run_model_sim(days=time1,
                                                                                                 population=dimension,
                                                                                                 case=case0,
                                                                                                 k_val=list_b_correct,
                                                                                                 b_val=b0,
                                                                                                 u0_val=list_u_correct[0],
                                                                                                 sigma_val=list_s_correct,
                                                                                                 Kmax0=der,
                                                                                                 a_val=list_Kmax_correct,
                                                                                                 c_val=c0, m_val=1,
                                                                                                 g_val=g0)

                                print('Running2')

                                # optimize

                                #Chemo or immuno need to switch the stament, because chemo can have a m value between 0 and 1, and immuno can only be 1 and 0
                                list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1, final = run_model_m(days=time1,
                                                                                                            population=dimension,
                                                                                                            case=case0,
                                                                                                            k_val=list_b_correct,
                                                                                                            b_val=b0,
                                                                                                            u0_val=
                                                                                                            list_u_correct[0],
                                                                                                            sigma_val=list_s_correct,
                                                                                                            Kmax0=der,
                                                                                                            a_val=list_Kmax_correct,
                                                                                                            c_val=c0,
                                                                                                            step_val=0.1,
                                                                                                            g_val=g0,
                                                                                                            obj='final',
                                                                                                            Chemo_or_Immuno= "Chemo", step_val_on = True)

                                print('Running3')

                                modelPredictions = list_x1
                                # print(modelPredictions)
                                # print(dimension- list_x)
                                # print('pred: ' + str(list_x))
                                if True:
                                    print('Plotting')
                                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                                    # fig.suptitle("Patient " + str(i+1) + ", tumor: "+ str(j+1), fontsize=16)#+ ". Optimal values of m: "+ str(m_vals[count]))
                                    ax0.plot(time1, np.array(list_Kmax1), label='m')
                                    ax0.legend(fontsize=14)
                                    ax0.set_xlabel("days from chemotherapy start", fontsize=16)
                                    ax0.set_ylabel("value of m", fontsize=16)
                                    ax1.plot(time1, list_u, label='u with constant treatment', color='black')
                                    ax1.plot(time1, list_u1, label='u with optimized treatment', color='red' )
                                    ax1.set_xlabel("days from treatment start", fontsize=16)
                                    ax1.set_ylabel("value of u", fontsize=16)
                                    ax2.set_xlabel("days from treatment start", fontsize=16)
                                    ax2.set_ylabel("volume of tumor", fontsize=16)
                                    ax1.legend(fontsize=14)
                                    ax2.plot(time1, list_x, label='x with constant treatment', color='black')
                                    ax2.plot(time1, list_x1, label="x with optimized treatment", color='red')
                                    ax2.legend(fontsize=14)

                                    # Create the folder if it doesn't exist.
                                    save_path = os.path.join(dataset_path, 'Results_nsclc_paper_optimized_treatment_Chemotherapy_fixed_plots', "PR_with_correct_sigma_chemo_step_on_integer_false",
                                                             f"{studyName}", "Optimization", functionToFit,
                                                             'Plots',
                                                             f"{arm}")  # added arm to it
                                    if not os.path.exists(save_path):
                                        os.makedirs(save_path)

                                    # Save the figure
                                    fig.savefig(os.path.join(save_path, str(key)))

                                    #fig.savefig(dataset_path + "OptimizationD/" + str(key))

                                    ###
                                    '''fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
                                    fig.suptitle("Exponential cost of resistance "  + arm, fontsize=12)
                                    ax1.scatter(time, list_u1, label='u', color='black', linestyle='dashed')
                                    ax1.legend( fontsize=12)
      
                                    ax1.set_title( " sigma=" +str(round(list_s, 3))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1)) 
                                    ax2.set_title("m: " + str(list_Kmax1))
                                    ax2.plot(time, dimension, label="real measurements", color='red')
                                    ax2.plot(time, list_x1, label='model predictions', color='blue')
                                    ax2.legend( fontsize=12)
                                    ax1.set_xlabel("days from immunotherapy treatment start", fontsize=12)
                                    ax1.set_ylabel("value of u", fontsize=12) 
                                    ax2.set_xlabel("days from immunotherapy treatment start", fontsize=12)
                                    ax2.set_ylabel("volume of tumor", fontsize=12)
                                    fig.savefig(dataset_path + "Optimization/" + str(key) )'''

                                '''except:
                                        print(key)
                                        result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                                                                                  values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn])
                                        continue'''

                                '''if len(set(dimension)) == 1:
                                    modelPredictions = dimension
                                else:
                                    modelPredictions =  list_x'''

                                modelPredictions = [0 if str(i) == 'nan' else i for i in modelPredictions]
                            # absError = modelPredictions - dimension
                            # SE = np.square(absError)
                            # temp_sum = np.sum(SE)
                            # MSE = np.mean(SE)

                            '''result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                                                                              values = [key, time, dimension, modelPredictions, abs(dimension[-1]- modelPredictions[-1]),
                                                                                        r2_score(dimension, modelPredictions), (2 * noParameters) - (2 * np.log(temp_sum)), group, cn]) #need to put parameter  mean_absolute_error(dimension, modelPredictions),'''
                        except:
                            continue


        #C, No write on dicts, so I dont think it is needed
        # # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        # a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")
        #
        # pickle.dump(result_dict, a_file)
        # a_file.close()

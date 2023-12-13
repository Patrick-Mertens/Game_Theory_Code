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
dataset_path = r"/home/bob/project_game_theory_linux/SpiderProject_KatherLab" # Use a raw string for the path
sys.path.insert(0,dataset_path)
paramControl = "4_November" #C, to many patients






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
# lim0= [20, 20, 20, 20, 20]
r_values = []
u_values = []
sigma_values = []
target = []
count = 0


WHO = 'PR_Params_K_neg_for_both_trend_sigma_0_by_k'

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

                        #firstDim = dimension[0:-3] #this was commented
                        #firstTime = time[0:-3]
                        firstDim = dimension #this was uncommented
                        firstTime = time
                        if 'MPDL3280A' in arm:  # or 'Docetaxel' in arm:
                            count += 1

                        try:





                            Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(
                                lim / 20, lim / 2, lim * 3, dimension, trend)

                            #C, no idea why
                            Fluctuate.extend(Evolution)  ##comment if 4 groupd
                            k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_chemo_vs_immuno(studyName,
                                                                                                                   arm,Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
                            # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all(dimension)

                            case0 = 'exp_K_neg'

                            list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed_unsaved(days=firstTime,
                                                                                                    population=firstDim,
                                                                                                    case=case0,
                                                                                                    k_val=k0,
                                                                                                    b_val=b0,
                                                                                                    u0_val=u0,
                                                                                                    sigma_val=sigma0,
                                                                                                    Kmax0=K0,
                                                                                                    a_val=a0,
                                                                                                    c_val=c0,
                                                                                                    free='k', #this was sigma
                                                                                                    g_val=g0)

                            print('run_model_fixed_unsaved working')
                            #Switching arm, to get alternative params
                            #arm = 'MPDL3280A'
                            #Turn this one if you want to go from chemo to immuno
                            #arm = 'DOCETAXEL'

                            # what would have happened to docetaxel if they had been given immuno
                            #k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_chemo_vs_immuno(studyName,arm,Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
                            # list_x1, list_u1 = list_x, list_u


                            ##Changing case for run_model sim:


                            list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1 = run_model_sim(days=time,
                                                                                                   population=dimension,
                                                                                                   case=case0,
                                                                                                   k_val=list_b,
                                                                                                   b_val=b0,
                                                                                                   u0_val=list_u[0],
                                                                                                   sigma_val=list_s,
                                                                                                   Kmax0=K0,
                                                                                                   a_val=list_Kmax,
                                                                                                   c_val=c0,
                                                                                                   m_val=1,
                                                                                                   g_val=g0)



                            print('run_model_sim is executed')
                            r_values.append(list_Kmax)
                            u_values.append(list_u[0])
                            sigma_values.append(list_s)
                            if list_x1[-1] > list_x[-1]:
                                target.append(0)
                            else:
                                target.append(1)
                            # optimize
                            # list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1, final =run_model_m(days=time, population=dimension, case=case0, k_val=list_b, b_val=b0, u0_val=list_u[0], sigma_val=list_s, Kmax0=K0, a_val=list_Kmax, c_val=c0, step_val=0, g_val=g0, obj='final')
                            # print(list_x1)
                            # print(time)
                            modelPredictions = list_x1


                            #if 'DOCETAXEL' in arm or 'Docetaxel' in arm:
                            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5),
                                                                     constrained_layout=True)
                            # fig.suptitle("Exponential cost of resistance "  + arm, fontsize=12)
                            ax1.plot(time, list_u, label='u', color='gray')
                            ax1.legend(fontsize=12)
                            # ax2.set_title( "Real tumor evolution under immunotherapy")
                            # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                            # ax2.set_title("m: " + str(list_Kmax1))
                            ax2.scatter(time, dimension, label="real measurements", color='red',marker='x')
                            ax2.plot(time, list_x, label='model predictions', color='blue')
                            ax2.legend(fontsize=12)
                            ax1.set_xlabel("days from treatment start", fontsize=12)
                            ax1.set_ylabel("value of u", fontsize=12)
                            ax2.set_xlabel("days from treatment start", fontsize=12)
                            ax2.set_ylabel("volume of tumor", fontsize=12)

                            ax3.plot(time, list_u1, label='u', color='gray')
                            ax3.legend(fontsize=12)
                            # ax4.set_title("Simulated tumor evolution under chemotherapy")
                            # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                            # ax2.set_title("m: " + str(list_Kmax1))
                            # ax2.scatter(time, dimension, label="real measurements", color='red',
                            #             marker='x')
                            ax4.scatter(time, dimension, label="real measurements", color='red',marker='x')
                            ax4.plot(time, list_x1, label='model predictions', color='blue')
                            ax4.legend(fontsize=12)
                            ax3.set_xlabel("days from treatment start", fontsize=12)
                            ax3.set_ylabel("value of u", fontsize=12)
                            ax4.set_xlabel("days from treatment start", fontsize=12)
                            ax4.set_ylabel("volume of tumor", fontsize=12)

                            # Create the folder if it doesn't exist.
                            save_path = os.path.join(dataset_path, 'Results_nsclc_paper_fighting_treatment' , WHO,
                                                     f"{studyName}", "Alt", functionToFit,
                                                     'Plots',f"{arm}",f"{trend}")  # added arm to it
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)

                            # Save the figure
                            fig.savefig(os.path.join(save_path, str(key)))
                                # fig.savefig(dataset_path + "Simulate immuno/" + str(key))

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
                            absError = abs(modelPredictions[-1] - dimension[-1])
                            SE = np.square(absError)
                            temp_sum = np.sum(SE)
                            MSE = np.mean(SE)

                            result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
                                                                     categories=['patientID', 'time', 'dimension',
                                                                                 'prediction', 'rmse', 'rSquare',
                                                                                 'aic', 'params', 'cancer'],
                                                                     values=[key, time, dimension, modelPredictions,
                                                                             abs(dimension[-1] - modelPredictions[
                                                                                 -1]),
                                                                             r2_score(dimension, modelPredictions),
                                                                             (2 * noParameters) - (
                                                                                         2 * np.log(temp_sum)),
                                                                             absError,
                                                                             cn])  # need to put parameter  mean_absolute_error(dimension, modelPredictions),
                        except:
                            continue

            # Properly join paths and check if directory exists
        full_Pickle_directory_path = os.path.join(dataset_path, 'Results_nsclc_paper_fighting_treatment', WHO,
                                                             f"{studyName}", "Alt", functionToFit,
                                                             'Pickle_Files',f"{arm}")

        # Check if directory exists and if not, create it
        if not os.path.exists(full_Pickle_directory_path):
            os.makedirs(full_Pickle_directory_path)

        # Now join the path for the file
        file_path = os.path.join(full_Pickle_directory_path, studyName + '.pkl')

        with builtins.open(file_path,"wb") as a_file:  # Using with ensures the file will be closed after the block
            pickle.dump(result_dict, a_file)

            a_file.close()
        # # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        # a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")
        #
        # pickle.dump(result_dict, a_file)
        # a_file.close()

#This is the secontion of the code that has the header """#opt""" until optimize
#In all honesty I have no clue yet what the function of this section is suppose dot be.
import os 
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from sklearn.metrics import r2_score

# copy dataset into working directory
dataset_path = 'SpiderProject_KatherLab'

from Utils import *
from VG_Functions import *
from FitFunctions import *

import pickle

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
    data, arms = Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']  # take only first tumor
    filtered_Data = filtered_Data.loc[
        filtered_Data['TRTESTCD'] == 'LDIAM']  # take only tumors for which measurement of longer diameter is available
    temp = list(filtered_Data['TRORRES'])  # this should be the measurements
    temp = Remove_String_From_Numeric_Vector(temp,
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
    data, arms = Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    for functionToFit in functions:

        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate', 'Evolution'],
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

                        time = Correct_Time_Vector(time, convertToWeek=True)

                        # If the value of Dimension is nan or any other string value, we replace it with zero
                        dimension = Remove_String_From_Numeric_Vector(dimension, valueToReplace=0)

                        dimension = [x for _, x in sorted(zip(time, dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy / maxi
                            # dimension_copy = dimension_copy/np.max(dimension_copy)

                        trend = Detect_Trend_Of_Data(dimension_copy)
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
                            if not ('DOCETAXEL' in arm or 'Docetaxel' in arm):
                                # Size1, Size2, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split_ind(lim/2, lim*2, dimension, trend)
                                # Size1, Size2,Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/lim0[sind], lim/2, lim*3, dimension, trend) #es lim*3
                                Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(
                                    lim / 20, lim / 2, lim * 3, dimension, trend)
                                '''if sind ==0 or sind ==1 or sind ==4:
                                  Size2= Size3 '''

                                Fluctuate.extend(Evolution)  ##comment if 4 groupd
                                k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName,
                                                                                                    dimension, arm)
                                # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all(dimension)

                                time1 = time + [time[-1] + 10, time[-1] + 20, time[-1] + 30, time[-1] + 40,
                                                time[-1] + 50]
                                # time1=time
                                # time1 = np.linspace(time[0], time[-1], 20, endpoint=True)

                                list_x0, list_u0, list_Kmax0, error0, list_b0, list_s0, der0 = run_model_fixed(
                                    days=time, population=firstDim, case=case0, k_val=k0, b_val=b0, u0_val=u0,
                                    sigma_val=sigma0, Kmax0=K0, a_val=a0, c_val=c0, free='k', g_val=g0)
                                # what would have happened to docetaxel if they had been given immuno
                                # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size( studyName, dimension, 'Docetaxel')

                                list_x, list_u, list_Kmax, error, list_b, list_s = run_model_sim(days=time1,
                                                                                                 population=dimension,
                                                                                                 case=case0,
                                                                                                 k_val=list_b0,
                                                                                                 b_val=b0,
                                                                                                 u0_val=list_u0[0],
                                                                                                 sigma_val=list_s0,
                                                                                                 Kmax0=K0,
                                                                                                 a_val=list_Kmax0,
                                                                                                 c_val=c0, m_val=1,
                                                                                                 g_val=g0)

                                # optimize

                                list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1, final = run_model_m(days=time1,
                                                                                                            population=dimension,
                                                                                                            case=case0,
                                                                                                            k_val=list_b0,
                                                                                                            b_val=b0,
                                                                                                            u0_val=
                                                                                                            list_u0[0],
                                                                                                            sigma_val=list_s0,
                                                                                                            Kmax0=K0,
                                                                                                            a_val=list_Kmax0,
                                                                                                            c_val=c0,
                                                                                                            step_val=0.1,
                                                                                                            g_val=g0,
                                                                                                            obj='final')

                                modelPredictions = list_x1
                                # print(modelPredictions)
                                # print(dimension- list_x)
                                # print('pred: ' + str(list_x))
                                if True:
                                    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
                                    # fig.suptitle("Patient " + str(i+1) + ", tumor: "+ str(j+1), fontsize=16)#+ ". Optimal values of m: "+ str(m_vals[count]))
                                    ax0.plot(time1, np.array(list_Kmax1), label='m')
                                    ax0.legend(fontsize=14)
                                    ax0.set_xlabel("days from immunotherapy start", fontsize=16)
                                    ax0.set_ylabel("value of m", fontsize=16)
                                    ax1.scatter(time1, list_u, label='u with constant treatment', color='black',
                                                linestyle='dashed')
                                    ax1.scatter(time1, list_u1, label='u with optimized treatment', color='red',
                                                linestyle='dashed')
                                    ax1.set_xlabel("days from treatment start", fontsize=16)
                                    ax1.set_ylabel("value of u", fontsize=16)
                                    ax2.set_xlabel("days from treatment start", fontsize=16)
                                    ax2.set_ylabel("volume of tumor", fontsize=16)
                                    ax1.legend(fontsize=14)
                                    ax2.plot(time1, list_x, label='x with constant treatment', color='black')
                                    ax2.plot(time1, list_x1, label="x with optimized treatment", color='red')
                                    ax2.legend(fontsize=14)
                                    fig.savefig(dataset_path + "OptimizationD/" + str(key))

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

        # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        #a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")

        #pickle.dump(result_dict, a_file)
        #a_file.close()

"""#simulate"""


def run_model_sim(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, m_val, g_val):
    list_x = []
    list_u = []
    list_Kmax = []
    list_b = []
    error = []
    der = []
    list_s = []

    # try:
    m = GEKKO(remote=False)
    eval = days
    # eval = np.linspace(days[i][j][0], days[i][j][-1], 20, endpoint=True)
    m.time = eval
    # disc= np.ones(len(days[i][j]))
    x_data = population
    x = m.Var(value=x_data[0], lb=0)
    sigma = m.Param(sigma_val)
    d = m.Param(c_val)
    k = m.Param(k_val)
    b = m.Param(b_val)
    r = m.Param(a_val)
    # step = [0 if z<0 else 1 for z in m.time]

    m_param = m.Param(m_val)
    u = m.Var(value=u0_val, lb=0)  # , ub=1)
    # m.free(u)
    a = m.Param(a_val)
    c = m.Param(c_val)
    Kmax = m.Param(Kmax0)

    if case == 'case4':
        m.Equations([x.dt() == x * (r * (1 - u ** 2) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-2 * u * r * (1 - x / (Kmax)) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'case0':
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (m_param * b / ((k + b * u) ** 2))])
    elif case == 'case3':
        m.Equations([x.dt() == (x) * (r * (1 - u) * (1 - x / Kmax) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (b * m_param / ((b * u + k) ** 2) - r * (1 - x / (Kmax)))])
    elif case == 'case5':
        m.Equations([x.dt() == x * (r * (1 + u ** 2) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (2 * u * r * (1 - x / (Kmax)) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'exp_r':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (m.exp(-g_val * u)) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-g_val * r * (1 - x / (Kmax)) * (m.exp(-g_val * u)) + (b * m_param) / (
                                 b * u + k) ** 2)])
    elif case == 'exp_K':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (
                                 (-g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'exp_both':
        m.Equations([x.dt() == x * (
                    r * (m.exp(-g_val * u)) * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-g_val * r * (m.exp(-g_val * u)) * (1 - x * (m.exp(g_val * u)) / (Kmax)) + (
                                 b * m_param) / ((b * u + k) ** 2) - g_val * r * x / (Kmax))])

    m.options.IMODE = 4
    m.options.SOLVER = 1
    m.options.NODES = 5  # collocation nodes

    # m.options.COLDSTART=2
    m.solve(disp=False, GUI=False)

    list_x = x.value
    list_u = u.value
    list_Kmax = m_param.value

    return list_x, list_u, list_Kmax, error, list_b, list_s


studies = ['a', 'a', 'c', 'd', 'e']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
splits = [True, True, False, True, True]
trends = ['Up', 'Down', 'Fluctuate']
studies = ['1', '2', '3', '4', '5']
# studies = ['3']
functions = ['Exponential']
###############################################################################

# FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions

maxList = []
minList = []

for studyName in studies:
    # rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path, 'Study_' + studyName + '_1.xlsx')
    sind = studies.index(studyName)
    sp = splits[sind]
    data, arms = Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    filtered_Data = data.loc[data['TRLINKID'] == 'INV-T001']  # take only first tumor
    filtered_Data = filtered_Data.loc[
        filtered_Data['TRTESTCD'] == 'LDIAM']  # take only tumors for which measurement of longer diameter is available
    temp = list(filtered_Data['TRORRES'])  # this should be the measurements
    temp = Remove_String_From_Numeric_Vector(temp,
                                                   valueToReplace=0)  # removes strings and replace by zero, why? do we only have strings when it disappears?
    # tempo = transform_to_volume(temp)
    maxList.append(max(temp))  # max value of measurement
    minList.append(min(temp))  # min value of measurement

###############################################################################

# Fit Funtions to the Data Points

# maxi = np.max([288, 0])
maxi = np.max(maxList)
studies = ['a', 'a', 'c', 'd', 'e']
studies = ['1', '2', '3', '4', '5']
studies = ['5']
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'GeneralGompertz']
functions = ['Exponential']
splits = [True, True, False, True, True]
# splits=[False]
noPars = [3, 3, 3, 4, 3, 4]

for studyName in studies:
    sind = studies.index(studyName)
    sp = splits[sind]
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True

    # rawDataPath = os.path.join(r"D:\Spider Project\rawData\new Files", studyName + '_m.xlsx')
    rawDataPath = os.path.join(dataset_path, 'Study_' + studyName + '_1.xlsx')
    data, arms = Read_Excel(rawDataPath, ArmName='TRT01A', split=sp)
    for functionToFit in functions:

        find = functions.index(functionToFit)
        noParameters = noPars[find]
        result_dict = Create_Result_dict(arms, ['Up', 'Down', 'Fluctuate'],
                                               categories=['patientID', 'rmse', 'rSquare',
                                                           'time', 'dimension', 'prediction', 'aic', 'params',
                                                           'cancer'])
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

                        time = Correct_Time_Vector(time, convertToWeek=True)

                        # If the value of Dimension is nan or any other string value, we replace it with zero
                        dimension = Remove_String_From_Numeric_Vector(dimension, valueToReplace=0)

                        dimension = [x for _, x in sorted(zip(time, dimension))]
                        dimension_copy = dimension.copy()
                        if normalizeDimension:
                            dimension_copy = dimension_copy / maxi
                            # dimension_copy = dimension_copy/np.max(dimension_copy)

                        trend = Detect_Trend_Of_Data(dimension_copy)

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

                        firstDim = dimension[0:-3]
                        firstTime = time[0:-3]
                        firstDim = dimension
                        firstTime = time

                        try:
                            Size1, Size2, Size4, Up, Down, Fluctuate = split1_ind(lim / 2, lim * 2, dimension, trend)
                            k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName, dimension,
                                                                                                arm)
                            list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=firstTime,
                                                                                                    population=firstDim,
                                                                                                    case=case0,
                                                                                                    k_val=k0, b_val=b0,
                                                                                                    u0_val=u0,
                                                                                                    sigma_val=sigma0,
                                                                                                    Kmax0=K0, a_val=a0,
                                                                                                    c_val=c0, free='k',
                                                                                                    g_val=g0)
                            list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1 = run_model_sim(days=time,
                                                                                                   population=dimension,
                                                                                                   case=case0,
                                                                                                   k_val=list_b,
                                                                                                   b_val=b0,
                                                                                                   u0_val=list_u[0],
                                                                                                   sigma_val=list_s,
                                                                                                   Kmax0=K0,
                                                                                                   a_val=list_Kmax,
                                                                                                   c_val=c0, m_val=1,
                                                                                                   g_val=g0)
                            modelPredictions = list_x1

                            '''except:
                                    print(key)
                                    result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                                                                              values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn])
                                    continue'''

                            if len(set(dimension)) == 1:
                                modelPredictions = dimension
                            else:
                                modelPredictions = list_x1

                            modelPredictions = [0 if str(i) == 'nan' else i for i in modelPredictions]
                            absError = modelPredictions - dimension
                            SE = np.square(absError)
                            temp_sum = np.sum(SE)
                            MSE = np.mean(SE)

                            result_dict = Write_On_Result_dict(result_dict, arm, trend,
                                                                     categories=['patientID', 'time', 'dimension',
                                                                                 'prediction', 'rmse', 'rSquare', 'aic',
                                                                                 'params', 'cancer'],
                                                                     values=[key, time, dimension, modelPredictions,
                                                                             mean_squared_error(dimension,
                                                                                                modelPredictions),
                                                                             r2_score(dimension, modelPredictions),
                                                                             (2 * noParameters) - (
                                                                                         2 * np.log(temp_sum)), group,
                                                                             cn])  # need to put parameter
                        except:
                            continue

        # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        #a_file = open(os.path.join(dataset_path, functionToFit, studyName + '.pkl'), "wb")

        #pickle.dump(result_dict, a_file)
        #a_file.close()

"""
Size1, Size2, Size4, Up, Down, Fluctuate = split1_ind(lim / 2, lim * 2, scaled_pop[i], list_trends[i])
dimension = scaled_pop[i]
time = scaled_days[i]
firstDim = dimension[0:-3]
firstTime = time[0:-3]
k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK(dimension)
list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=firstTime, population=firstDim, case=case0,
                                                                        k_val=k0, b_val=b0, u0_val=u0, sigma_val=sigma0,
                                                                        Kmax0=K0, a_val=a0, c_val=c0, free='sigma',
                                                                        g_val=g0)
list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1 = run_model_sim(days=time, population=dimension, case=case0,
                                                                       k_val=k0, b_val=b0, u0_val=list_u[0],
                                                                       sigma_val=list_s, Kmax0=K0, a_val=list_Kmax,
                                                                       c_val=c0, m_val=1, g_val=g0)
list_x
"""

###############################################################################
# Plot HeatMaps
###############################################################################

result = pd.DataFrame()
for f in functions:
    temp = []
    indices = []
    for s in studies:
        # result_dict = pickle.load( open( r"D:\Spider Project\Fit\080221\\" + f + '\\' + s + ".pkl", "rb" ) )
        #result_dict = pickle.load(open(dataset_path + f + '/' + s + ".pkl", "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                indices.append(arm + '_' + trend)
                temp.append(np.around(np.nanmean(result_dict[arm][trend]['rSquare']), 3))
    result[f] = temp

result.index = indices
result.dropna(inplace=True)
minValuesObj = result.min(axis=1)

tab_n = result.div(result.max(axis=1), axis=0)
#cmap = sns.cm.rocket
#mpl.rcParams['font.size'] = 10
#plt.rcParams["font.weight"] = "bold"
#plt.rcParams["axes.labelweight"] = "bold"
# plt.figure()
#plt.tight_layout()
# t = tab_n.T
#ax = sns.heatmap(tab_n, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
#ax.set_xticklabels(labels=functions, rotation=30, fontsize=10)
#plt.title('R-Squared values for each arms', fontsize=20)

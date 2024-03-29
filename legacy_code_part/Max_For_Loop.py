#C, this part of the code are 3 naxinum for loops, both start with:
#FIND MAXIMUM OF THE DATA SETS To be able to Normaliz the Whole Tumor Dimensions
#Thus I believe that each one is used for each type of tumor or maybe something akin like that
#For that reason I need to take a better look at the code.

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
studies = ['1', '2', '3', '4', '5']
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
# limits =[ 0.003316135059100506

r_values = []
u_values = []
sigma_values = []
initial_x = []
initial_trend = []
target = []

for studyName in studies:
    sind = studies.index(studyName)
    sp = splits[sind]
    studyName = studies[sind]
    warnings.filterwarnings("ignore")
    normalizeDimension = True
    lim = limits[sind]
    lim = 0.003316135059100506  # depende de los grupos

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

                        # firstDim = dimension[0:-3]
                        # firstTime = time[0:-3]

                        try:
                            # if True:
                            Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim / 20,
                                                                                                              lim / 2,
                                                                                                              lim * 3,
                                                                                                              dimension,
                                                                                                              trend)
                            # Size1, Size2,Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/lim0[sind], lim/2, lim*3, dimension, trend) #es lim*3
                            '''if sind ==0 or sind ==1 or sind ==4:
                              Size2= Size3 '''

                            Fluctuate.extend(Evolution)  ##comment if 4 groupd
                            k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName, dimension,
                                                                                                arm)
                            # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m(dimension)

                            list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=time,
                                                                                                    population=dimension,
                                                                                                    case=case0,
                                                                                                    k_val=k0, b_val=b0,
                                                                                                    u0_val=u0,
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
                                target.append(1)  # 1 means success
                            modelPredictions = list_x
                            # print(dimension- list_x)
                            # print('pred: ' + str(list_x))
                            if trend == 'Evolution':
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                                fig.suptitle("Exponential cost of resistance " + arm, fontsize=12)
                                ax1.scatter(time, list_u, label='u', color='black', linestyle='dashed')
                                ax1.legend(fontsize=12)

                                # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                                # ax2.set_title("m: " + str(list_Kmax1))
                                ax2.plot(time, dimension, label="real measurements", color='red')
                                ax2.plot(time, list_x, label='model predictions', color='blue')
                                ax2.legend(fontsize=12)
                                ax1.set_xlabel("days from treatment start", fontsize=12)
                                ax1.set_ylabel("value of u", fontsize=12)
                                ax2.set_xlabel("days from treatment start", fontsize=12)
                                ax2.set_ylabel("volume of tumor", fontsize=12)
                                fig.savefig(dataset_path + "Sigma 0/" + str(key))

                            '''except:
                            print(key)
                            result_dict =  utils.Write_On_Result_dict(result_dict, arm, trend, categories = ['patientID','time', 'dimension', 'prediction', 'rmse', 'rSquare','aic', 'params', 'cancer'], 
                                                                      values = [key, time, dimension, np.nan, np.nan,np.nan, np.nan, np.nan, cn])
                            continue'''

                            if len(set(dimension)) == 1:
                                modelPredictions = dimension
                            else:
                                modelPredictions = list_x

                            modelPredictions = [0 if str(i) == 'nan' else i for i in modelPredictions]
                            absError = modelPredictions - dimension
                            SE = np.square(absError)
                            temp_sum = np.sum(SE)
                            MSE = np.mean(SE)
                            '''print(time)  
                            print(dimension)
                            print(modelPredictions)
                            print(mean_squared_error(dimension, modelPredictions))
                            print(r2_score(dimension, modelPredictions))'''

                            result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
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
        a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")

        pickle.dump(result_dict, a_file)
        a_file.close()

result_dict

result_dict = utils.Write_On_Result_dict(result_dict, arm, trend,
                                         categories=['patientID', 'time', 'dimension', 'prediction', 'rmse', 'rSquare',
                                                     'aic', 'params', 'cancer'],
                                         values=[key, time, dimension, modelPredictions,
                                                 mean_squared_error(dimension, modelPredictions),
                                                 r2_score(dimension, modelPredictions),
                                                 (2 * noParameters) - (2 * np.log(temp_sum)), group, cn])
result_dict

"""#plots"""

###############################################################################
# Plot HeatMaps
###############################################################################
import math

trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
result1 = pd.DataFrame()
for f in functions:
    temp = []
    indices = []
    for s in studies:
        # result_dict = pickle.load( open( r"D:\Spider Project\Fit\080221\\" + f + '\\' + s + ".pkl", "rb" ) )
        result_dict = pickle.load(open(dataset_path + f + '/' + s + ".pkl", "rb"))

        arms = list(result_dict.keys())
        arms.sort()
        for arm in arms:
            for trend in trends:
                indices.append(arm + '_' + trend)
                temp.append(np.around(np.nanmean(result_dict[arm][trend]['rSquare']), 3))  # rmse
                # temp.append(((np.around(np.nanmedian(result_dict[arm][trend]['params']), 6)))) #rmse

    result1[f] = temp

result1.index = indices
result1.dropna(inplace=True)
minValuesObj = result1.min(axis=1)

tab_n = result1.div(result1.max(axis=1), axis=0)
cmap = sns.cm.rocket
mpl.rcParams['font.size'] = 10
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
# plt.figure()
plt.tight_layout()
# t = tab_n.T
ax = sns.heatmap(tab_n, cmap=sns.color_palette("rocket", as_cmap=True), xticklabels=True, yticklabels=True, square=True)
# ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
ax.set_xticklabels(labels=functions, rotation=30, fontsize=10)
plt.title('R-Squared values for each arms', fontsize=20)

for f in functions:
    temp = []
    indices = []
    for s in studies:
        # result_dict = pickle.load( open( r"D:\Spider Project\Fit\080221\\" + f + '\\' + s + ".pkl", "rb" ) )
        result_dict = pickle.load(open(dataset_path + f + '/' + s + ".pkl", "rb"))
result_dict

result1.columns = ['Game theoretical']

"""#simulation"""

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
list_arms = []
list_trends = []

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

                        firstDim = dimension[0:-3]
                        firstTime = time[0:-3]
                        # firstDim = dimension
                        # firstTime = time
                        if 'MPDL3280A' in arm:  # or 'Docetaxel' in arm:
                            count += 1

                        try:
                            # if True:
                            # if 'DOCETAXEL' in arm or 'Docetaxel' in arm:
                            # if 'MPDL3280A' in arm:
                            # count+=1
                            # Size1, Size2, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split_ind(lim/2, lim*2, dimension, trend)
                            # Size1, Size2,Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim/lim0[sind], lim/2, lim*3, dimension, trend) #es lim*3
                            Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec = split1_ind(lim / 20,
                                                                                                              lim / 2,
                                                                                                              lim * 3,
                                                                                                              dimension,
                                                                                                              trend)
                            '''if sind ==0 or sind ==1 or sind ==4:
                              Size2= Size3 '''

                            Fluctuate.extend(Evolution)  ##comment if 4 groupd
                            k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName, dimension,
                                                                                                arm)
                            # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all(dimension)

                            list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=firstTime,
                                                                                                    population=firstDim,
                                                                                                    case=case0,
                                                                                                    k_val=k0, b_val=b0,
                                                                                                    u0_val=u0,
                                                                                                    sigma_val=sigma0,
                                                                                                    Kmax0=K0, a_val=a0,
                                                                                                    c_val=c0,
                                                                                                    free='sigma',
                                                                                                    g_val=g0)

                            # what would have happened to docetaxel if they had been given immuno
                            # k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size( studyName, dimension, 'MPDL3280A')
                            # list_x1, list_u1 = list_x, list_u

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
                            r_values.append(list_Kmax)
                            u_values.append(list_u[0])
                            sigma_values.append(list_s)
                            # if list_x1[-1] > list_x[-1] and list_x1[-1]>list_x1[0]:
                            # target.append(0)
                            if list_x1[-1] < list_x[-1]:
                                target.append(trend)
                                list_arms.append(arm)
                            list_trends.append(trend)

                            # optimize
                            # list_x1, list_u1, list_Kmax1, error1, list_b1, list_s1, final =run_model_m(days=time, population=dimension, case=case0, k_val=list_b, b_val=b0, u0_val=list_u[0], sigma_val=list_s, Kmax0=K0, a_val=list_Kmax, c_val=c0, step_val=0, g_val=g0, obj='final')
                            # print(list_x1)
                            # print(time)
                            modelPredictions = list_x1
                            # print(modelPredictions)
                            # print(dimension- list_x)
                            # print('pred: ' + str(list_x))
                            if True:
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
                                # fig.suptitle("Exponential cost of resistance "  + arm, fontsize=12)
                                ax1.scatter(time, list_u1, label='u', color='black', linestyle='dashed')
                                ax1.legend(fontsize=12)

                                # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                                # ax2.set_title("m: " + str(list_Kmax1))
                                ax2.plot(time, dimension, label="real measurements", color='red')
                                ax2.plot(time, list_x1, label='model predictions', color='blue')
                                ax2.legend(fontsize=12)
                                ax1.set_xlabel("days from treatment start", fontsize=12)
                                ax1.set_ylabel("value of u", fontsize=12)
                                ax2.set_xlabel("days from treatment start", fontsize=12)
                                ax2.set_ylabel("volume of tumor", fontsize=12)
                                fig.savefig(dataset_path + "alternative prediction/" + str(key))

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
                                                                                 'prediction', 'rmse', 'rSquare', 'aic',
                                                                                 'params', 'cancer'],
                                                                     values=[key, time, dimension, modelPredictions,
                                                                             abs(dimension[-1] - modelPredictions[-1]),
                                                                             r2_score(dimension, modelPredictions),
                                                                             (2 * noParameters) - (
                                                                                         2 * np.log(temp_sum)),
                                                                             absError,
                                                                             cn])  # need to put parameter  mean_absolute_error(dimension, modelPredictions),
                        except:
                            continue

        # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")

        pickle.dump(result_dict, a_file)
        a_file.close()

c_up = 0
c_down = 0
c_ev = 0
c_fluct = 0
target = list_trends
for i in range(len(target)):
    if target[i] == 'Up':
        c_up += 1
    elif target[i] == 'Down':
        c_down += 1
    elif target[i] == 'Evolution':
        c_ev += 1
    else:
        c_fluct += 1
print(c_up)
print(c_down)
print(c_ev)
print(c_fluct)
# 63 ,72, 14, 80 0.78, 0.5, 0.7, 0.45

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

                        # firstDim = dimension[0:-3]
                        # firstTime = time[0:-3]
                        firstDim = dimension
                        firstTime = time
                        if 'MPDL3280A' in arm:  # or 'Docetaxel' in arm:
                            count += 1

                        try:

                            if 'DOCETAXEL' in arm or 'Docetaxel' in arm:
                                # if 'MPDL3280A' in arm:
                                # count+=1
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

                                list_x, list_u, list_Kmax, error, list_b, list_s, der = run_model_fixed(days=firstTime,
                                                                                                        population=firstDim,
                                                                                                        case=case0,
                                                                                                        k_val=k0,
                                                                                                        b_val=b0,
                                                                                                        u0_val=u0,
                                                                                                        sigma_val=sigma0,
                                                                                                        Kmax0=K0,
                                                                                                        a_val=a0,
                                                                                                        c_val=c0,
                                                                                                        free='sigma',
                                                                                                        g_val=g0)

                                # what would have happened to docetaxel if they had been given immuno
                                k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size(studyName,
                                                                                                    dimension,
                                                                                                    'MPDL3280A')
                                # list_x1, list_u1 = list_x, list_u

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
                                # print(modelPredictions)
                                # print(dimension- list_x)
                                # print('pred: ' + str(list_x))
                                if 'DOCETAXEL' in arm or 'Docetaxel' in arm:
                                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5),
                                                                             constrained_layout=True)
                                    # fig.suptitle("Exponential cost of resistance "  + arm, fontsize=12)
                                    ax1.scatter(time, list_u, label='u', color='black', linestyle='dashed')
                                    ax1.legend(fontsize=12)
                                    # ax2.set_title( "Real tumor evolution under immunotherapy")
                                    # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                                    # ax2.set_title("m: " + str(list_Kmax1))
                                    ax2.plot(time, dimension, label="real measurements", color='red')
                                    ax2.plot(time, list_x, label='model predictions', color='blue')
                                    ax2.legend(fontsize=12)
                                    ax1.set_xlabel("days from treatment start", fontsize=12)
                                    ax1.set_ylabel("value of u", fontsize=12)
                                    ax2.set_xlabel("days from treatment start", fontsize=12)
                                    ax2.set_ylabel("volume of tumor", fontsize=12)

                                    ax3.scatter(time, list_u1, label='u', color='black', linestyle='dashed')
                                    ax3.legend(fontsize=12)
                                    # ax4.set_title("Simulated tumor evolution under chemotherapy")
                                    # ax1.set_title( " sigma=" +str(round(list_s, 0))  + " b= " + str(b0) + " K= " + str(K0) +", u0: " + str(round(list_u[0],3))+ " r: " + str(round(list_Kmax,3)))# + "m: " + str(list_Kmax1))
                                    # ax2.set_title("m: " + str(list_Kmax1))
                                    ax4.plot(time, dimension, label="real measurements", color='red')
                                    ax4.plot(time, list_x1, label='model predictions', color='blue')
                                    ax4.legend(fontsize=12)
                                    ax3.set_xlabel("days from treatment start", fontsize=12)
                                    ax3.set_ylabel("value of u", fontsize=12)
                                    ax4.set_xlabel("days from treatment start", fontsize=12)
                                    ax4.set_ylabel("volume of tumor", fontsize=12)
                                    fig.savefig(dataset_path + "Simulate immuno/" + str(key))

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

        # a_file = open(os.path.join(r"D:\Spider Project\Fit\080221", functionToFit, studyName + '.pkl'), "wb")
        a_file = open(os.path.join(dataset_path + functionToFit, studyName + '.pkl'), "wb")

        pickle.dump(result_dict, a_file)
        a_file.close()

c_up = 0
c_down = 0
c_ev = 0
c_fluct = 0
for i in range(len(target)):
    if target[i] == 1:
        c_up += 1
    elif target[i] == 0:
        c_down += 1

print(c_up)
print(c_down)
print(c_ev)
print(c_fluct)
# 63 ,72, 14, 80 0.7

xyz = pd.DataFrame({"r": r_values,
                    "u": u_values, "sigma": sigma_values})
Target = pd.DataFrame({"target": target})
data = [xyz, Target]
# headers = ["x", "trend", "target"]
df = pd.concat(data, axis=1)
Target['target'].value_counts()

from sklearn.utils import resample

df_majority = df[df.target == 0]
df_minority = df[df.target == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=80,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
df_upsampled.target.value_counts()

df_upsampled

from sklearn.model_selection import train_test_split

xyz = df_upsampled[['r', 'u', 'sigma']]
Target = df_upsampled[['target']]
x_train, x_test = train_test_split(xyz, test_size=0.3, random_state=42, shuffle=True)
y_train, y_test = train_test_split(Target, test_size=0.3, random_state=42, shuffle=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)

prediction = (clf.predict(x_train))
clf.score(x_test, y_test)

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(x_train, y_train)
prediction = (clf.predict(x_test))

clf.score(x_test, y_test)

(prediction)

pd.set_option("display.max_rows", None, "display.max_columns", None)

print(y_test)

y_test = (y_test.to_numpy())

len(y_train)
# success 17 de 59

succes = 0
for i in range(len(y_test)):
    if y_test[i] == 1:
        succes += 1
    print(y_test[i] - prediction[i])
print(succes)

r_values = np.array(r_values)
u_values = np.array(u_values)
sigma_values = np.array(sigma_values)
target = np.array(target)
len(target)

import scipy  # correlation between initial volume or trend and target

print(scipy.stats.kendalltau(r_values, target))
print(scipy.stats.kendalltau(u_values, target))
print(scipy.stats.kendalltau(sigma_values, target))

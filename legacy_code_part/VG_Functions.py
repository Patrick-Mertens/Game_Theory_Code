#C, this part start at the coment """#Mine""" where mine probably refers to Virginia
#code line 373, but this script will be broken where needed.

from math import pi
import statistics

#C, this pacakge is defined above, new functions so, I think it is used during it.
from gekko import GEKKO
import math
#C, this pacakge is defined above, new functions so, I think it is used during it.
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statistics


def transform_to_volume(list_diameters0):
    patient = []
    for j in range(len(list_diameters0)):
        r = list_diameters0[j] / 2
        patient.append((4 / 3) * pi * r ** 3)

    return patient


def split1(min_val, med_val, max_val, scaled_pop0):  # this is for the list of lists
    size1 = []
    size2 = []
    size3 = []
    size4 = []
    Inc = []
    Dec = []
    for i in range(len(scaled_pop0)):
        # for j in range(len(scaled_days0[i])):
        if (scaled_pop0[i][0]) < min_val:
            size1.append((i))
        elif (scaled_pop0[i][0]) < med_val:
            size2.append((i))
        elif (scaled_pop0[i][0]) < max_val:
            size3.append((i))
        else:
            size4.append((i))
    for i in range(len(scaled_pop0)):
        # for j in range(len(scaled_days[i])):
        if scaled_pop0[i][0] > scaled_pop0[i][1]:
            Dec.append((i))
        else:
            Inc.append((i))
    return size1, size2, size3, size4, Inc, Dec


def split1_ind(min_val, med_val, max_val, scaled_pop0,
               trend0):  # this is for determining a single element (so input is a single list corresponding to a tumor)
    size1 = []
    size2 = []
    size3 = []
    size4 = []
    Inc = []
    Dec = []
    Fluctuate = []
    Evolution = []
    Up = []
    Down = []
    if (scaled_pop0[0]) < min_val:
        size1.append(scaled_pop0)
    elif (scaled_pop0[0]) < med_val:
        size2.append(scaled_pop0)
    elif (scaled_pop0[0]) < max_val: #C, I assume that this is not used in OPT.py, bcs size3 is not defined.
        size3.append(scaled_pop0)
    else:
        size4.append(scaled_pop0)

    if scaled_pop0[0] > scaled_pop0[1]:
        Dec.append(scaled_pop0)
    else:
        Inc.append(scaled_pop0)
    if trend0 == 'Up':
        Up.append(scaled_pop0)
    elif trend0 == 'Down':
        Down.append(scaled_pop0)
    elif trend0 == 'Fluctuate':
        Fluctuate.append(scaled_pop0)
    elif trend0 == 'Evolution':
        Evolution.append(scaled_pop0)
    if scaled_pop0[0] > scaled_pop0[1]:
        Dec.append(scaled_pop0)
    else:
        Inc.append(scaled_pop0)
    return size1, size2, size3, size4, Up, Down, Fluctuate, Evolution, Inc, Dec


def split_ind(min_val, max_val, scaled_pop0,
              trend0):  # this is for determining a single element (so input is a single list corresponding to a tumor)
    size1 = []
    size2 = []
    size3 = []
    size4 = []
    Inc = []
    Dec = []
    Fluctuate = []
    Evolution = []
    Up = []
    Down = []
    if (scaled_pop0[0]) < min_val:
        size1.append(scaled_pop0)
    elif (scaled_pop0[0]) < max_val:
        size2.append(scaled_pop0)
    else:
        size4.append(scaled_pop0)

    if scaled_pop0[0] > scaled_pop0[1]:
        Dec.append(scaled_pop0)
    else:
        Inc.append(scaled_pop0)
    if trend0 == 'Up':
        Up.append(scaled_pop0)
    elif trend0 == 'Down':
        Down.append(scaled_pop0)
    elif trend0 == 'Fluctuate':
        Fluctuate.append(scaled_pop0)
    elif trend0 == 'Evolution':
        Evolution.append(scaled_pop0)
    if scaled_pop0[0] > scaled_pop0[1]:
        Dec.append(scaled_pop0)
    else:
        Inc.append(scaled_pop0)
    return size1, size2, size4, Up, Down, Fluctuate, Evolution, Inc, Dec


def limit(scaled_days):
    first = []
    for i in range(len(scaled_days)):
        first.append(scaled_days[i][0])
    return statistics.median(first)


def limit1(scaled_days):
    first = []
    for i in range(len(scaled_days)):
        for j in range(len(scaled_days[i])):
            first.append(scaled_days[i][j][0])
    return statistics.median(first)

# Size1, Size2, Size4, Inc, Dec = split1(0.0005, 0.004, scaled_days, scaled_pop)


def Detect_Trend_Of_Data(vector):
    diff = []
    for d in range(len(vector) - 1):
        diff.append(vector[d + 1] - vector[
            d])  # compute the difference between each 2 measurements, kind of like the slope in each intervañ
    s_pos = 0
    for x in diff:
        if x > 0:
            s_pos = s_pos + x  # if it is increasing, add the difference of measurements (so it measures the amount of centimeters that tumor grows)

    s_neg = 0
    for x in diff:
        if x < 0:
            s_neg = s_neg + x  # measures how much tumor decreases in total

    if all(i >= 0 for i in diff):  # if al intervals are increasing, Up
        trend = 'Up'
    elif all(i <= 0 for i in diff):  # if all intervals are decreasing, down
        trend = 'Down'
    elif diff[0] < 0 and (vector[-1] > vector[0] or (diff[-1] > -diff[0] / 2)) and (
            max(vector) == vector[0] or max(vector) == vector[-1]):
        trend = 'Evolution'
    # elif diff[0]>0 and vector[-1]<= vector[0]:
    #   trend = 'Delayed'
    else:
        trend = 'Fluctuate'

    '''elif vector[0]< max(vector) and vector[-1]< max(vector):
        trend = 'Delayed'

    elif vector[0] >min(vector) and vector[-1] >min(vector) and vector[-1]> vector[0]/2:
        trend = 'Evolution'
    else:
      trend = 'Fluctuate'
    elif diff[0] > 0 and not abs(s_neg) >= (s_pos /2): #if amount they decrease is smaller than half of the amount it increases, Up
        trend = 'Up'
    elif diff[0] < 0 and not s_pos >= (abs(s_neg) /2):
        trend = 'Down'
    else:
        trend = 'Fluctuate'''
    return trend


def run_model_fixed(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, free, g_val=0.5):
    list_x = []
    list_u = []
    list_Kmax = []
    list_b = []
    error = []
    der = []
    list_s = []
    # try:
    m = GEKKO(remote=False)
    m.time = days
    x_data = population
    x = m.CV(value=x_data, lb=0);
    x.FSTATUS = 1  # fit to measurement
    x.SPLO = 0
    if free == 'sigma':
        # sigma= m.Param(0)
        sigma = m.FV(value=0.01, lb=0, ub=0.1);
        sigma.STATUS = 1
        # sigma = m.FV(value=0.01, lb= 0, ub=1); sigma.STATUS=1

        k = m.Param(k_val)
    elif free == 'k':
        sigma = m.Param(0)
        k = m.Param(k_val)

        # k = m.FV(value=0.1, lb= 0, ub=10); k.STATUS=1
    d = m.Param(c_val)
    b = m.Param(b_val)
    # g_val = m.FV(value=0.5, lb= 0); g_val.STATUS=1

    r = m.FV(value=0.4, lb=c_val, ub=1);
    r.STATUS = 1  # , ub=a_val  #estaba en 0.01 y lb= 0.00001
    # r = m.FV(value=c_val, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

    # r = m.FV(value=0.01, lb=0.00001, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

    step = [0 if z < 0 else 1 for z in m.time]
    m_param = m.Param(1)
    u = m.Var(value=u0_val, lb=0)  # , ub=1)
    m.free(u)
    a = m.Param(a_val)
    c = m.Param(c_val)
    Kmax = m.Param(Kmax0)
    if case == 'case3':
        m.Equations([x.dt() == (x) * (r * (1 - u) * (1 - x / Kmax) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (b * m_param / ((b * u + k) ** 2) - r * (1 - x / (Kmax)))])
    elif case == 'case0':
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (m_param * b / ((k + b * u) ** 2))])
    elif case == 'case4':
        m.Equations([x.dt() == x * (r * (1 - u ** 2) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-2 * u * r * (1 - x / (Kmax)) + (b * m_param) / (b * u + k) ** 2)])
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
    elif case == 'exp_K_benefit':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (
                                 (g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])

    elif case == 'exp_both':
        m.Equations([x.dt() == x * (
                    r * (m.exp(-g_val * u)) * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-g_val * r * (m.exp(-g_val * u)) * (1 - x * (m.exp(g_val * u)) / (Kmax)) + (
                                 b * m_param) / ((b * u + k) ** 2) - g_val * r * x / (Kmax))])
    elif case == 'exp':
        m.Equations([x.dt() == x * (r * (1 - u) * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-r * (1 - (x * m.exp(g_val * u)) / Kmax) - (
                                 g_val * r * x * (1 - u) * m.exp(g_val * u)) / Kmax + (b * m) / (b * u + k) ** 2)])
    elif case == 'exp_K_neg':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == -sigma * (
                                 (-g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])

    m.options.IMODE = 5  # dynamic estimation
    m.options.NODES = 5  # collocation nodes
    m.options.EV_TYPE = 2  # linear error (2 for squared)
    # m.options.MAX_ITER=15000

    m.solve(disp=False, debug=False)  # do not display solver output
    list_x = x.value
    list_u = u.value
    der = Kmax.value[0]
    list_Kmax = r.value[0]
    list_b = k.value[0]
    list_s = sigma.value[0]
    '''except: 
    list_x =0
    list_u= 0
    der = 0
    list_Kmax= 0
    list_b = 0
    list_s = 0'''
    return list_x, list_u, list_Kmax, error, list_b, list_s, der


# function to
def gridsearch(days, pop, model, k_vals, b_vals, cases, u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, g_vals):
    list_mse = []
    mse0 = 10000000
    # mse0=0

    Dict = {'mse': 10000000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'a_val': 0, 'c_val': 0,
            'g_val': 0}
    for k_val in k_vals:
        for Kmax_val in Kmax_vals:
            for a_val in a_vals:
                for c_val in c_vals:
                    for b_val in b_vals:
                        for case in cases:
                            for u0_val in u0_vals:
                                for sigma_val in sigma_vals:
                                    for g_val in g_vals:
                                        list_x, list_u, list_Kmax, error, list_b, list_s, der = model(days, pop, case,
                                                                                                      k_val, b_val,
                                                                                                      u0_val, sigma_val,
                                                                                                      Kmax_val, a_val,
                                                                                                      c_val, 'sigma',
                                                                                                      g_val)
                                        if mse(pop, list_x, error) == float(mse(pop, list_x, error)) and mse(pop,
                                                                                                             list_x,
                                                                                                             error) < mse0:
                                            # if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) > mse0:

                                            mse0 = mse(pop, list_x, error)
                                            Dict['mse'] = mse0
                                            Dict['k_val'] = k_val
                                            Dict['Kmax0'] = Kmax_val
                                            Dict['b_val'] = b_val
                                            Dict['case'] = case
                                            Dict['u0_val'] = u0_val
                                            Dict['sigma_val'] = sigma_val
                                            Dict['a_val'] = a_val
                                            Dict['c_val'] = c_val
                                            Dict['g_val'] = g_val
                                        list_mse.append(mse(pop, list_x, error))
    return Dict


def mse(x_true, x_pred, error):
    list_mse = []
    # for i in range(len(x_true)):
    # for j in range(len(x_true[i])):
    # if (i) not in error:
    # if not( True in np.isnan(np.array(x_pred[i]))):
    list_mse.append(mean_absolute_error(x_true, x_pred))
    # list_mse.append(r2_score(x_true, x_pred))
    return statistics.mean(list_mse)


def get_error(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val):
    list_dict1 = []
    bad = 0
    for i in range(len(scaled_pop)):
        # for j in range(len(list_days[i])):
        if (i) in group and (i) in response:
            try:

                D = gridsearch(days=scaled_days[i], pop=scaled_pop[i], model=run_model_fixed, k_vals=[k_val],
                               b_vals=[b_val], cases=[case], u0_vals=[u0_val], sigma_vals=[sigma_val],
                               Kmax_vals=[Kmax_val], a_vals=[a_val], c_vals=[c_val], g_vals=[g_val])
                list_dict1.append(D)
            except:
                list_dict1.append(
                    {'mse': 1000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'sigma_val': 0})
    good = 0
    list_error = []
    for count in range(len(list_dict1)):
        if (
        list_dict1[count]['mse']) < 0.1:  # and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
            good += 1
        if list_dict1[count]['mse'] != 1000:
            list_error.append(list_dict1[count]['mse'])
        else:
            bad += 1
    # Added by RACHEL as crashed here due to trying to take mean of empty list
    if (len(list_error) < 1):
        print("Empty list of errors!")
        m = 0
    else:
        m = statistics.mean(list_error)

    return good, m, len(list_dict1) - good - bad


def get_param(size, response):
    errors = []
    Dict = {'mse': 1000, 'k_val': 0, 'b_val': 0, 'case': 0, 'good': 0, 'u0_val': 0, 'K': 0, 'a': 0, 'c': 0, 'g': 0,
            'sigma': 0}
    for k in [0.2, 0.9, 2, 1, 5]:
        for b in [0.2, 1, 2, 10, 20]:
            for case in ['exp_K']:
                for u0 in [0.1]:
                    for sigma in [0]:
                        for K in [1, 2]:
                            for a_val in [1]:
                                for c_val in [0.001, 0.01, 0.1, 0.2]:
                                    for g_val in [0.1, 0.5, 0.9]:  # 0.1, 0.5, 0.9
                                        results = (
                                            get_error(size, response, k, b, case, u0, sigma, K, a_val, c_val, g_val))
                                        # if results[0] >Dict['good'] or (results[0]== Dict['good'] and results[1]< Dict['mse']):
                                        if results[1] < Dict['mse']:
                                            Dict['mse'] = results[1]
                                            Dict['k_val'] = k
                                            Dict['b_val'] = b
                                            Dict['case'] = case
                                            Dict['good'] = results[0]
                                            Dict['u0_val'] = u0
                                            Dict['sigma'] = sigma
                                            Dict['K'] = K
                                            Dict['a'] = a_val
                                            Dict['c'] = c_val
                                            Dict['g'] = g_val

    return Dict['mse'], Dict['k_val'], Dict['b_val'], Dict['case'], Dict['u0_val'], Dict['sigma'], Dict['K'], Dict['a'], \
    Dict['c'], Dict['g']


"""#simulate"""


def run_model_sim(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, m_val, g_val):
    list_x = []
    list_u = []
    list_Kmax = []
    list_b = []
    error = []
    der = []
    list_s = []

    m = GEKKO(remote=False)
    # eval= days[i][j]
    eval = days
    # eval = np.linspace(days[i][j][0], days[i][j][-1], 20, endpoint=True)
    m.time = eval
    # disc= np.ones(len(days[i][j]))
    # x_data= population[i][j]
    x_data = population
    x = m.Var(value=x_data[0], lb=0)
    sigma = m.Param(sigma_val)
    d = m.Param(c_val)
    k = m.Param(k_val)
    b = m.Param(b_val)
    r = m.Param(a_val)
    step = [0 if z < 0 else 1 for z in m.time]

    m_param = m.Param(m_val)
    u = m.Var(value=u0_val, lb=0)
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
    elif case == 'exp_K':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (
                                 (-g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'exp_K_neg':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == -sigma * (
                                 (-g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])

    m.options.IMODE = 4
    m.options.SOLVER = 1
    m.options.NODES = 5  # collocation nodes

    # m.options.COLDSTART=2
    m.solve(disp=False, GUI=False)

    list_x = x.value
    # list_Kmax.append(m_param.value)
    list_u = u.value
    # list_b.append(b_val)
    # list_s.append(sigma_val)

    return list_x, list_u, list_Kmax, error, list_b, list_s


def separate_by_size_predict_newdata4k_expK_all_m(tuple0):  # free sigma for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        9.602616255922462e-05, 2, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.4924284769001023e-05, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)



    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005581028152651971, 2, 0.2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0002474685799481297, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008712237364711401, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00045665782397184303, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)


    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.003805638444789154, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00583392900567394, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_m2(tuple0):  # free sigma for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        4.9007374519872045e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        6.46287476594794e-06, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)



    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000208892569995844, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.4680954119098068e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)


    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008186625558637999, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0003065002240221485, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)


    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0020903478069908797, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0017453868791226252, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_d(tuple0):  # free sigma for all dataset with docetaxel! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.4973845374143e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.001913325901334e-05, 0.2, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.629359474403977e-05, 1, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        6.240565937342348e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005069629697468392, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00028086457826024576, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)


    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.003322438544321211, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0024268534223027454, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


"""#sigma = 0"""


def separate_by_size_predict_newdata4k_expK_all_m(tuple0):  # free k for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        7.313143992107043e-05, 5, 10, 'exp_K', 0.1, 0, 2, 1, 0.1, 0.9)
        # (8.275172474111321e-05, 2, 0.2, 'exp_K', 0.1, 0, 2, 1, 0.01, 0.9)
        # (0.000305984404353992, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.0869733859844166e-05, 5, 20, 'exp_K', 0.1, 0, 2, 1, 0.01, 0.9)
    # (2.1380076175407215e-05, 2, 20, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.1)
    # (3.090507410772475e-05, 0.2, 0.2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)

    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005799316839025912, 5, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.2, 0.1)
        # (0.0006452113104681421, 2, 2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.5)
        # (0.0015723368019712983, 0.2, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00027987108205535595, 2, 1, 'exp_K', 0.1, 0, 1, 1, 0.2, 0.9)
    # (0.0002927863727123914, 2, 2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.00038168865371151817, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)

    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (0.0011137926237382563, 5, 1, 'exp_K', 0.1, 0, 1, 1, 0.2, 0.9)
        # (0.0011671916939567178, 2, 2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.1)
        # (0.001458928940703507, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000678070085753398, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.5)
    # (0.000678070085753398, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.5)
    # (0.000978074804508313, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)

    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004425836264998339, 5, 0.2, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.9)
        # (0.00450514460723673, 2, 0.2, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.9)
        # (0.004621358245057331, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.007104403461273823, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.007104403461273823, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.007564481322379863, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_m2(tuple0):  # free k for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00030737542219077434, 2, 10, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.5)
        # (0.00041895133169086703, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.3778734629797172e-05, 2, 0.2, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.1)
    # (1.8539896481844664e-05, 0.2, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.5)

    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00017661728403719335, 2, 2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.5)
        # (0.00042335960940577947, 0.2, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        4.0865313358232235e-05, 2, 10, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.9)
    # (0.0001214330333908684, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0009299596062593311, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.001, 0.9)
        # (0.000990566657551813, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00044743471357201, 0.9, 2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.0004977257829369905, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)

    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0024470026491540823, 0.2, 10, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
        # (0.0024470026491540823, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0026884887576161046, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.003079177227178332, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_d(tuple0):  # free k for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.515522486541801e-05, 2, 20, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.9)
        # (9.949456399194937e-05, 0.2, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        9.84988726540814e-06, 2, 2, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.5)
    # (3.49558641280033e-05, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.1)

    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00011189208709435016, 2, 20, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.1)
        # (0.00037159335527561923, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00018089711071741444, 2, 1, 'exp_K', 0.1, 0, 2, 1, 0.01, 0.1)
    # (0.0002401936154066152, 0.2, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)

    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0007740678242624623, 0.9, 0.2, 'exp_K', 0.1, 0, 2, 1, 0.001, 0.5)
        # (0.0008188476838805763, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005776449046505248, 2, 1, 'exp_K', 0.1, 0, 1, 1, 0.01, 0.9)
    # (0.0008186204142754958, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004249380085258772, 0.9, 20, 'exp_K', 0.1, 0, 2, 1, 0.01, 0.1)
        # (0.004261832992533772, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.1)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004137171032429125, 2, 0.2, 'exp_K', 0.1, 0, 1, 1, 0.001, 0.9)
    # (0.005301902658870665, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


"""#free k"""


def separate_by_size_predict_newdata4k_expK_all_m2(tuple0):  # free k for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.318073100164206e-05, 0.2, 1, 'exp_K', 0.1, 0.1, 2, 0.7, 0.01, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.1988993513864182e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.5)



    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000170864179072142, 0.2, 20, 'exp_K', 0.1, 0.1, 1, 0.7, 0.2, 0.5)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        4.3696923116113385e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.5)


    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008208047636721348, 0.2, 2, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00036691469333449754, 0.2, 10, 'exp_K', 0.1, 0.1, 2, 0.7, 0.2, 0.5)


    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.002145623289254397, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0019104266503177422, 0.2, 1, 'exp_K', 0.1, 0.1, 1, 0.7, 0.2, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


# def separate_by_size_predict_newdata4k_expK_all_d(tuple0):  # free k for all dataset with docetaxel! 4 size groups
#     k0 = 0
#     b0 = 0
#     case0 = 'c'
#     group = 's'
#     K0 = 2
#     r = 0
#     sigma = 0
#     u0 = 0
#     a0 = 0
#     c0 = 0
#     K0 = 0
#     g0 = 0
#     if tuple0 in Size1 and tuple0 in Inc:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         8.242930323913428e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.9)
#
#         group = 'Size1, Inc'
#     elif tuple0 in Size1 and tuple0 in Dec:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         8.87496562253954e-06, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.5)
#
#
#     elif tuple0 in Size2 and tuple0 in Inc:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         8.54868897899777e-05, 0.2, 20, 'exp_K', 0.1, 0.1, 2, 0.7, 0.2, 0.9)
#
#         group = 'Size2, Inc'
#     elif tuple0 in Size2 and tuple0 in Dec:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         8.95258530253879e-05, 0.2, 1, 'exp_K', 0.1, 0.1, 2, 0.7, 0.2, 0.9)
#
#
#     elif tuple0 in Size3 and tuple0 in Inc:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         0.0007489938071873247, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)
#
#         group = 'Size2, Inc'
#     elif tuple0 in Size3 and tuple0 in Dec:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         0.0003549248265986936, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.1)
#
#
#
#     elif tuple0 in Size4 and tuple0 in Inc:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         0.0035659552321450037, 0.2, 2, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.5)
#
#         group = 'Size3, Inc'
#     elif tuple0 in Size4 and tuple0 in Dec:
#         error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
#         0.00276126528512829, 0.2, 1, 'exp_K', 0.1, 0.1, 2, 0.7, 0.2, 0.9)
#
#     return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0
#

def separate_by_size_predict_newdata4k_expK_all_m(tuple0):  # free k for all dataset with MDA! 4 size groups
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.914549773022699e-05, 0.2, 1, 'exp_K', 0.1, 0.1, 1, 0.7, 0.2, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.0594130952867914e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.9)


    elif tuple0 in Size2 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005084367676080137, 0.2, 20, 'exp_K', 0.1, 0.1, 1, 0.7, 0.2, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00027976317971715125, 0.2, 2, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.9)

    elif tuple0 in Size3 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008728845870223612, 0.2, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size3 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005105128432293237, 0.2, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.9)


    elif tuple0 in Size4 and tuple0 in Inc:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0036588720460737426, 0.2, 1, 'exp_K', 0.1, 0.1, 2, 0.7, 0.01, 0.9)

        group = 'Size3, Inc'
    elif tuple0 in Size4 and tuple0 in Dec:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.005885976549904761, 0.2, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.01, 0.9)

    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


"""#separate by medication"""


def separate_by_size_predict_newdata4k_expK_all(tuple0):  # free sigma forall dataset! ()
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005165071147921052, 1, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.3255574155391976e-05, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00013707286889543799, 1, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00013707286889543799, 1, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)

        group = 'Size1, Evol'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0006666468537480382, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00018372458165303473, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0010037019907797348, 1, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size2, Fluc'
    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00034507009809860124, 1, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size2, Evol'
    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0047240928696678, 1, 20, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.1)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004457598111540281, 0.5, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.9)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.003360149061068511, 1, 2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size3, Fuct'
    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0017028117475238011, 0.5, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.9)

        group = 'Size4, Evol'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK4(tuple0):  # free sigma for dataset 4! (immunotherapy)
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000404826742117313, 1, 0.9, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.947790844406902e-05, 1, 2, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005496881110342027, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.5)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000764052433139788, 1, 2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.5)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0001757729059128098, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.1)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0009949343062233824, 1, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.5)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0029976375840254465, 2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0076979236778310116, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.1)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.003377343735265962, 1, 2, 'exp_K', 0.1, 0.01, 2, 1, 0.1, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK4d(tuple0):  # free sigma for dataset 4! (docetaxel)
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.4191015257504047e-05, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.1)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.3432637676043346e-05, 2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.795856667796206e-05, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.403882940428784e-05, 0.5, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.70128931881017e-05, 1, 10, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.1)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005466638220975943, 0.2, 10, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0011959778208764003, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0012613771216098855, 0.5, 10, 'exp_K', 0.1, 0.01, 1, 1, 0.1, 0.9)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.003140959823264545, 0.5, 10, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.1)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK3n(tuple0):  # free sigma for dataset 3! (non squamos)
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.237264630057773e-05, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.4200526966979503e-05, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.1)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        6.680240892724842e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.1, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0002863235679044799, 0.5, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        8.075105822507534e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005014541662187393, 0.5, 2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.5)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008849428959647237, 1, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.1, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0018898633427355306, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.9)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0011592051002556618, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK3(tuple0):  # free sigma for dataset 3! (squamos)
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00026866054535390553, 1, 2, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.1)
        # (4.567635332464182e-15, 2, 0.9, 'exp_K', 0.1, 0.01, 10000, 1, 0.1, 0.5)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.4949481895727995e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.5)
        # (2.431745320706542e-06, 0.5, 0.9, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.9)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        5.2944007919463655e-05, 1, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.9)
        # (1.7635536233435253e-05, 0.5, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008295087358736322, 2, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.5)
        # (0.0014466381966600158, 1, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00011541597087205478, 1, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.5)
        # (5.705154954749632e-05, 1, 0.9, 'exp_K', 0.1, 0.01, 10000, 1, 0.1, 0.9)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0010519493545457675, 1, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.1, 0.9)
        # (0.0003182327965691808, 2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.1, 0.5)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0013843791379245642, 0.5, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.9)
        # (0.0005611217688855046, 1, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0003008142451465958, 0.5, 2, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)
        # (0.00019863607799175425, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)
        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.002776983856942883, 2, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

        # (0.0034263005054791067, 1, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


"""#4 grupos"""


def separate_by_size_predict_newdata4k_expK3(tuple0):  # free sigma for dataset 3!  Fluct and Evol
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00023322078370586707, 0.5, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.1)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.4445425193128378e-05, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.1)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        6.7994688882105e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.9)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        3.2609695341588405e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0005746119350296415, 0.5, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        9.161412023452672e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)
        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0006641093701929936, 2, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)
    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00017762442979236132, 0.5, 2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.001143874609843865, 1, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.05, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.001435031417128963, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.001904880129781463, 2, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0013545197485380555, 0.5, 5, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK5(tuple0):  # free sigma for dataset 5!  Fluct and Evol
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00021913131358899105, 0.5, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.7491597134332922e-05, 2, 5, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0001536980406935319, 0.5, 5, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00024540402641491914, 0.2, 2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0006116237730980924, 2, 5, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.1)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.914558106383703e-05, 2, 0.2, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0011012379964550397, 2, 10, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00046776836878047824, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.5)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00818880547060987, 0.5, 0.2, 'exp_K', 0.1, 0.01, 10000, 1, 0.05, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0016394485058577233, 1, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0021800041653283276, 2, 5, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.5)

    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.006811868082787102, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK4(tuple0):  # free sigma for dataset 4!  Fluct and Evol
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000370647903944737, 1, 0.9, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        3.157293149283325e-05, 1, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.5)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00047837803369333914, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.5)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00010291615822316393, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.5)
        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0007345426715830883, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00014363701803459246, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.05, 0.1)
        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0007770740723934302, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)
    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0008727785084622991, 0.2, 10, 'exp_K', 0.1, 0.01, 1, 1, 0.05, 0.9)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0025998901402886215, 2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004812258100195442, 0.5, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0036592019628764894, 1, 10, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0023614977388961775, 1, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK2(tuple0):  # free sigma for dataset 2!  Fluct and Evol
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0006189047689710229, 2, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        2.3879427450384092e-05, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.02, 0.5)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00016373899274394274, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.02, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00010502455417579896, 1, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.1)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0004609238736646776, 1, 2, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.5)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.000285984479975258, 1, 2, 'exp_K', 0.1, 0.01, 1, 1, 0.02, 0.5)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00045143530817454624, 1, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.5)
    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0002405844877023254, 0.5, 20, 'exp_K', 0.1, 0.01, 1.5, 1, 0.02, 0.9)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.004041722652713276, 0.2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0036261036724600745, 2, 0.2, 'exp_K', 0.1, 0.01, 1, 1, 0.02, 0.9)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00346831319757535, 2, 0.9, 'exp_K', 0.1, 0.01, 2, 1, 0.02, 0.9)

    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.002609367584837647, 0.5, 2, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK1(tuple0):  # free sigma for dataset 1!  Fluct and Evol
    k0 = 0
    b0 = 0
    case0 = 'c'
    group = 's'
    K0 = 2
    r = 0
    sigma = 0
    u0 = 0
    a0 = 0
    c0 = 0
    K0 = 0
    g0 = 0
    if tuple0 in Size1 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00029347289730922136, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.02, 0.1)

        group = 'Size1, Inc'
    elif tuple0 in Size1 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        4.3803522362540425e-06, 0.5, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.02, 0.5)

        group = 'Size1, Dec'
    elif tuple0 in Size1 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        4.412913789831186e-05, 0.5, 10, 'exp_K', 0.1, 0.01, 1.5, 1, 0.02, 0.5)

        group = 'Size1, Fluc'
    elif tuple0 in Size1 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        1.800585772354985e-05, 1, 0.2, 'exp_K', 0.1, 0.01, 1.5, 1, 0.01, 0.9)

        group = 'Size1, Fluc'
    elif tuple0 in Size2 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0002492541826179748, 0.5, 0.2, 'exp_K', 0.1, 0.01, 2, 1, 0.01, 0.9)

        group = 'Size2, Inc'
    elif tuple0 in Size2 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0002121124757156387, 0.5, 10, 'exp_K', 0.1, 0.01, 1.5, 1, 0.02, 0.9)

        group = 'Size2, Dec'
    elif tuple0 in Size2 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00046950591138649405, 1, 20, 'exp_K', 0.1, 0.01, 2, 1, 0.02, 0.9)
    elif tuple0 in Size2 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.00023953953171746879, 1, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

        group = 'Size2, Fluc'

    elif tuple0 in Size4 and tuple0 in Up:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.005024555234610955, 2, 0.9, 'exp_K', 0.1, 0.01, 1, 1, 0.01, 0.9)

        group = 'Size3, Inc'

    elif tuple0 in Size4 and tuple0 in Down:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0018223177836747646, 0.2, 5, 'exp_K', 0.1, 0.01, 1.5, 1, 0.02, 0.5)

        group = 'Size3, Dec'

    elif tuple0 in Size4 and tuple0 in Fluctuate:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0019117045282755036, 1, 0.9, 'exp_K', 0.1, 0.01, 10000, 1, 0.01, 0.5)
    elif tuple0 in Size4 and tuple0 in Evolution:
        error, k0, b0, case0, u0, sigma, K0, a0, c0, g0 = (
        0.0017093963948650182, 0.2, 10, 'exp_K', 0.1, 0.01, 10000, 1, 0.02, 0.9)

        group = 'Size3, Fuct'
    return k0, b0, group, case0, u0, sigma, K0, a0, c0, g0


"""#results"""


def separate_by_size(study, pop, arm):
    if arm == 'DOCETAXEL' or arm == 'docetaxel':
        k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_d(pop)
    elif arm == 'MPDL3280A':
        k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m(pop)
    else:
        print(f"hello world")
        k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m2(pop)

    return k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0


# def separate_by_size(study, pop, arm):
#     if arm == 'DOCETAXEL' or arm == 'docetaxel':
#         k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_d(pop)
#     else:
#         k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m(pop)
#
#     return k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0

"""#optimize"""
#C, I think that this is the model that we need to have.
def run_model_m(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, step_val, g_val, obj):
    list_x = []
    list_u = []
    list_Kmax = []
    list_b = []
    error = []
    der = []
    list_s = []
    m = GEKKO(remote=False)
    eval = days
    # eval = np.linspace(days[0], days[-1], 20, endpoint=True)
    m.time = eval
    x = m.Var(value=population[0], lb=0)
    sigma = m.Param(sigma_val)
    d = m.Param(c_val)
    k = m.Param(k_val)
    b = m.Param(b_val)
    r = m.Param(a_val)
    step = np.random.normal(0.5, 0.5, len(eval))
    step = np.ones(len(eval))
    # step= step_val*step
    step[0] = 1
    m_param = m.MV(value=step, lb=0, ub=1, integer=True);
    m_param.STATUS = 1
    u = m.Var(value=u0_val, lb=0)
    a = m.Param(a_val)
    c = m.Param(c_val)
    Kmax = m.Param(Kmax0)

    if case == 'case3':
        m.Equations([x.dt() == (x) * (r * (1 - u) * (1 - x / Kmax) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (b * m_param / ((b * u + k) ** 2) - r * (1 - x / (Kmax)))])
    elif case == 'case0':
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (m_param * b / ((k + b * u) ** 2))])
    elif case == 'case4':
        m.Equations([x.dt() == x * (r * (1 - u ** 2) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (-2 * u * r * (1 - x / (Kmax)) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'case5':
        m.Equations([x.dt() == x * (r * (1 + u ** 2) * (1 - x / (Kmax)) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (2 * u * r * (1 - x / (Kmax)) + (b * m_param) / (b * u + k) ** 2)])
    elif case == 'exp_K':
        # u unbounded for this one
        m.Equations([x.dt() == x * (r * (1 - x / (Kmax * (m.exp(-g_val * u)))) - m_param / (k + b * u) - d),
                     u.dt() == sigma * (
                                 (-g_val * r * x * (m.exp(g_val * u))) / (Kmax) + (b * m_param) / (b * u + k) ** 2)])

    p = np.zeros(len(eval))
    p[-1] = 1.0
    final = m.Param(value=p)
    if obj == 'final':
        m.Obj(x * final)
    elif obj == 'x':
        m.Obj(x)
    elif obj == 'variance':
        p = np.ones(len(eval))
        # mean = sum(x.value)/(len(eval))
        m.Obj(((x.value) * p - (sum(x.value)) / (len(eval))) ** 2)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    m.options.NODES = 5  # collocation nodes

    # optimize
    m.solve(disp=False, GUI=False)
    m.options.OBJFCNVAL

    list_x = x.value
    list_u = u.value
    # list_der.append(Kmax.value[0])
    list_Kmax = m_param.value

    return list_x, list_u, list_Kmax, error, list_b, list_s, m.options.OBJFCNVAL

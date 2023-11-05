#This contains the self written functions from the nsclc_paper.ipyn
import builtins
from math import pi
import statistics
from gekko import GEKKO
import math
import numpy as np
from datetime import datetime
import sys as sys
import os
import csv
import traceback
#import cupy as cp



def transform_to_volume(list_diameters0):
  patient=[]
  for j in range(len(list_diameters0)):
    r = list_diameters0[j]/2
    patient.append((4/3)* pi * r**3)

  return patient


def split1(min_val,med_val,max_val, scaled_pop0): #this is for the list of lists
  size1=[]
  size2=[]
  size3=[]
  size4=[]
  Inc=[]
  Dec=[]
  for i in range(len(scaled_pop0)):
    #for j in range(len(scaled_days0[i])):
      if (scaled_pop0[i][0])<min_val:
        size1.append((i))
      elif (scaled_pop0[i][0])<med_val:
        size2.append((i))
      elif (scaled_pop0[i][0])<max_val:
        size3.append((i))
      else:
        size4.append((i))
  for i in range(len(scaled_pop0)):
    #for j in range(len(scaled_days[i])):
      if scaled_pop0[i][0]> scaled_pop0[i][1]:
        Dec.append((i))
      else:
        Inc.append((i))
  return size1, size2, size3, size4, Inc, Dec

def split1_ind(min_val,med_val,max_val, scaled_pop0, trend0): #this is for determining a single element (so input is a single list corresponding to a tumor)
  size1=[]
  size2=[]
  size3=[]
  size4=[]
  Inc=[]
  Dec=[]
  Fluctuate=[]
  Evolution=[]
  Up=[]
  Down=[]
  if (scaled_pop0[0])<min_val:
    size1.append(scaled_pop0)
  elif (scaled_pop0[0])<med_val:
    size2.append(scaled_pop0)
  elif (scaled_pop0[0])<max_val:
    size3.append(scaled_pop0)
  else:
    size4.append(scaled_pop0)

  if scaled_pop0[0]> scaled_pop0[1]:
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
  if scaled_pop0[0]> scaled_pop0[1]:
      Dec.append(scaled_pop0)
  else:
      Inc.append(scaled_pop0)
  return size1, size2, size3, size4, Up, Down, Fluctuate, Evolution, Inc, Dec

def PR_split1_ind(min_val,med_val,max_val, scaled_pop0, trend0): #this is for determining a single element (so input is a single list corresponding to a tumor)
  size1=[]
  size2=[]
  size3=[]
  size4=[]
  Inc=[]
  Dec=[]
  Fluctuate=[]
  Evolution=[]
  Up=[]
  Down=[]
  if (scaled_pop0[0])<min_val:
    size1.append(scaled_pop0)
  elif (scaled_pop0[0])<med_val:
    size2.append(scaled_pop0)
  elif (scaled_pop0[0])<max_val:
    size3.append(scaled_pop0)
  else:
    size4.append(scaled_pop0)

  if scaled_pop0[0]> scaled_pop0[1]:
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
  # if scaled_pop0[0]> scaled_pop0[1]:
  #     Dec.append(scaled_pop0)
  # else:
  #     Inc.append(scaled_pop0)
  return size1, size2, size3, size4, Up, Down, Fluctuate, Evolution, Inc, Dec



def split_ind(min_val,max_val, scaled_pop0, trend0): #this is for determining a single element (so input is a single list corresponding to a tumor)
  size1=[]
  size2=[]
  size3=[]
  size4=[]
  Inc=[]
  Dec=[]
  Fluctuate=[]
  Evolution=[]
  Up=[]
  Down=[]
  if (scaled_pop0[0])<min_val:
    size1.append(scaled_pop0)
  elif (scaled_pop0[0])<max_val:
    size2.append(scaled_pop0)
  else:
    size4.append(scaled_pop0)

  if scaled_pop0[0]> scaled_pop0[1]:
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
  if scaled_pop0[0]> scaled_pop0[1]:
      Dec.append(scaled_pop0)
  else:
      Inc.append(scaled_pop0)
  return size1, size2, size4, Up, Down, Fluctuate, Evolution, Inc, Dec

def limit(scaled_days):
  first=[]
  for i in range(len(scaled_days)):

    first.append(scaled_days[i][0])

  print(f"Smallest number in the list: {min(first)}")
  print(f"Largest number in the list: {max(first)}")
  return statistics.median(first)

def limit1(scaled_days):
  first=[]
  for i in range(len(scaled_days)):
    for j in range(len(scaled_days[i])):
      first.append(scaled_days[i][j][0])
  return statistics.median(first)

#Size1, Size2, Size4, Inc, Dec = split1(0.0005, 0.004, scaled_days, scaled_pop)

def scale_data( list_pop0, max_diameter):
  scaled_pop=[]
  for i in range(len(list_pop0)):
    scaled = ((list_pop0[i] - 0)/(max_diameter - 0))
    scaled_pop.append(scaled)
  return scaled_pop

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


# i want that if it decreases at the beginning and then it increases more than  first point, it is evolved and if it increases

def run_model_fixed(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, free, g_val, ID): #C, g_val used to have default value of g=0.5
  # print("Run_model_fixed is runn")
  # # Printing the values and types in days
  # print("\n--- days ---")
  # print(days)
  # print(f"Total Count: {len(days) if isinstance(days, list) else 'Not a list'}")
  # print(f"Type: {type(days)}")
  #
  # # Printing the values and types in population
  # print("\n--- population ---")
  # print(population)
  # print(f"Total Count: {len(population) if isinstance(population, list) else 'Not a list'}")
  # print(f"Type: {type(population)}")
  #
  # # Printing the values and types in case
  # print("\n--- case ---")
  # print(case)
  # print(f"Type: {type(case)}")
  #
  # # Printing the values and types in k_val
  # print("\n--- k_val ---")
  # print(k_val)
  # print(f"Type: {type(k_val)}")
  #
  # # Printing the values and types in b_val
  # print("\n--- b_val ---")
  # print(b_val)
  # print(f"Type: {type(b_val)}")
  #
  # # Printing the values and types in u0_val
  # print("\n--- u0_val ---")
  # print(u0_val)
  # print(f"Type: {type(u0_val)}")
  #
  # # Printing the values and types in sigma_val
  # print("\n--- sigma_val ---")
  # print(sigma_val)
  # print(f"Type: {type(sigma_val)}")
  #
  # # Printing the values and types in Kmax0
  # print("\n--- Kmax0 ---")
  # print(Kmax0)
  # print(f"Type: {type(Kmax0)}")
  #
  # # Printing the values and types in a_val
  # print("\n--- a_val ---")
  # print(a_val)
  # print(f"Type: {type(a_val)}")
  #
  # # Printing the values and types in c_val
  # print("\n--- c_val ---")
  # print(c_val)
  # print(f"Type: {type(c_val)}")
  #
  # # Printing the values and types in free
  # print("\n--- free ---")
  # print(free)
  # print(f"Type: {type(free)}")
  #
  # # Printing the values and types in g_val
  # print("\n--- g_val ---")
  # print(g_val)
  # print(f"Type: {type(g_val)}")

  # # Printing the values and types in ID
  # print("\n--- ID ---")
  # print(ID)
  # print(f"Type: {type(ID)}")

  # # Printing the values and types in Check
  # print("\n--- Check ---")
  # print(Check)
  # print(f"Type: {type(Check)}")
  # print("END OF RUN MODEL DEBUGGING")

  list_x =[]
  list_u =[]
  list_Kmax =[]
  list_b =[]
  error=[]
  der=[]
  list_s=[]
  #try:
  m = GEKKO(remote=False)
  m.time= days
  x_data= population
  x = m.CV(value=x_data, lb=0); x.FSTATUS = 1 # fit to measurement
  x.SPLO = 0
  if free =='sigma':
    #sigma= m.Param(0)
    sigma = m.FV(value=0.01, lb= 0, ub=0.1); sigma.STATUS=1
    #sigma = m.FV(value=0.01, lb= 0, ub=1); sigma.STATUS=1

    k = m.Param(k_val)
  elif free== 'k':
    sigma= m.Param(0)
    k = m.Param(k_val)

    #k = m.FV(value=0.1, lb= 0, ub=10); k.STATUS=1
  d = m.Param(c_val)
  b = m.Param(b_val)
  #g_val = m.FV(value=0.5, lb= 0); g_val.STATUS=1


  r = m.FV(value=0.4, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001
  #r = m.FV(value=c_val, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  #r = m.FV(value=0.01, lb=0.00001, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  step = [0 if z<0 else 1 for z in m.time]
  m_param = m.Param(1)
  u = m.Var(value=u0_val, lb=0)#, ub=1)
  m.free(u)
  a = m.Param(a_val)
  c= m.Param(c_val)
  Kmax= m.Param(Kmax0)
  if case == 'case3':
    m.Equations([x.dt()==  (x)*(r*(1-u)*(1-x/Kmax)-m_param/(k+b*u)-d), u.dt()==sigma*(b*m_param/((b*u+k)**2)-r*(1-x/(Kmax)))])
  elif case == 'case0':
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt()== sigma*(m_param*b/((k+b*u)**2))])
  elif case == 'case4':
    m.Equations([x.dt()==  x*(r*(1-u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'case5':
    m.Equations([x.dt()==  x*(r*(1+u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_r':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(1-x/(Kmax))*(m.exp(-g_val*u))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K':

    # print(f"right statement, exp_K block_Fixed")
    # # u unbounded for this one
    # print("\n------Debuging equation------")
    # print(f"\nEntering case 'exp_K'")
    #
    # # Printing the variables and their types before they are used
    # print(f"g_val: {g_val}, Type: {type(g_val)}")
    # print(f"u: {u}, Type: {type(u)}")
    # print(f"r: {r}, Type: {type(r)}")
    # print(f"x: {x}, Type: {type(x)}")
    # print(f"Kmax: {Kmax}, Type: {type(Kmax)}")
    # print(f"m_param: {m_param}, Type: {type(m_param)}")
    # print(f"k: {k}, Type: {type(k)}")
    # print(f"b: {b}, Type: {type(b)}")
    # print(f"d: {d}, Type: {type(d)}")
    # print("\n ---- END STATEMENT -----")

    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K_benefit':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  elif case == 'exp_both':
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(m.exp(-g_val*u))*(1-x*(m.exp(g_val*u))/(Kmax))+(b*m_param)/((b*u+k)**2)-g_val*r*x/(Kmax))])
  elif case == 'exp':
    m.Equations([x.dt()==  x*(r*(1-u)*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-r*(1-(x*m.exp(g_val*u))/Kmax)-(g_val*r*x*(1-u)*m.exp(g_val*u))/Kmax+(b*m)/(b*u+k)**2)])
  elif case == 'exp_K_neg':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == -sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  m.options.IMODE = 5  # dynamic estimation
  m.options.NODES = 5   # collocation nodes
  m.options.EV_TYPE = 2 # linear error (2 for squared)
  #m.options.MAX_ITER=15000

  m.solve(disp=False, debug=False)    # do not display solver output
  list_x =x.value
  list_u= u.value
  der = Kmax.value[0]
  list_Kmax= r.value[0]
  list_b = k.value[0]
  list_s = sigma.value[0]
  '''except:
  list_x =0
  list_u= 0
  der = 0
  list_Kmax= 0
  list_b = 0
  list_s = 0'''

  ##Getting stuff
  formatted_time = datetime.now().strftime("%d_%H_%M_%S")

  # Setting path
  dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"
  sys.path.insert(0, dataset_path)

  target_dir = os.path.join(dataset_path, "Edited", "Model_params","Run_Singles_Run_model_fixed","Index_kept",f"{ID}", f"k_val-{k_val}",f"b_val-{b_val}",f"Kmax0-{Kmax0}",f"c_val-{c_val}",f"g_val-{g_val}")
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)



  # Using the function to print the information of variables
  print(f"\n--------- RUN_MODEL_FIXED_after_equation check-------------------")
  print(f"PatientID: {ID}")
  print(f"Time: {formatted_time}")


  print_variable_info('list_x', list_x)
  print_variable_info('list_u', list_u)
  print_variable_info('list_Kmax', list_Kmax)
  print_variable_info('error', error)
  print_variable_info('list_b', list_b)
  print_variable_info('list_s', list_s)
  print_variable_info('der', der)

  variables = [list_x, list_u, list_Kmax, error, list_b, list_s, der]
  lengths = []

  for var in variables:
    var_type = type(var)
    if isinstance(var, list) or var_type == 'gekko.gk_operators.GK_Value':
      lengths.append(len(var))
    elif isinstance(var, (int, float)):
      # We won't add this to the lengths list as it's a single value
      pass
    else:
      print(f"Warning: Unhandled type {type(var)} for variable {var}")

  # Determining max length
  if lengths:  # Check if lengths is not empty to avoid ValueError
    max_length = max(lengths)
  else:
    print("Error: No iterable variables to determine max length.")
    max_length = 0

  # Using your extend_to_length function to make all lists the same length, bcs dont know if it is all different lengths
  extended_list_x = extend_to_length(list_x, max_length)
  extended_list_u = extend_to_length(list_u, max_length)
  extended_list_Kmax = extend_to_length(list_Kmax, max_length)
  extended_error = extend_to_length(error, max_length)
  extended_list_b = extend_to_length(list_b, max_length)
  extended_list_s = extend_to_length(list_s, max_length)
  extended_der = extend_to_length(der, max_length)
  extended_ID = extend_to_length(ID, max_length)

  # Save to CSV
  filename = os.path.join(target_dir, f"{formatted_time}" + ".csv")
  with builtins.open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID", "list_x", "list_u", "list_Kmax", "error", "list_b", "list_s", "der"])
    writer.writerows(
      zip(extended_ID, extended_list_x, extended_list_u, extended_list_Kmax, extended_error, extended_list_b,
          extended_list_s,
          extended_der))

  print("Modelis complete run (run_model_fixed)")
  return list_x, list_u, list_Kmax, error, list_b, list_s, der


#function to
def gridsearch(days, pop, model, k_vals,  b_vals, cases,u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, g_vals, ID):
  list_mse=[]
  mse0=10000000
  #mse0=0

  Dict = {'mse':10000000, 'Kmax0':0, 'k_val' : 0, 'b_val':0, 'case': 0,  'u0_val':0, 'a_val':0, 'c_val':0, 'g_val':0 }
  for k_val in k_vals:
    for Kmax_val in Kmax_vals:
      for a_val in a_vals:
        for c_val in c_vals:
          for b_val in b_vals:
            for case in cases:
              for u0_val in u0_vals:
                for sigma_val in sigma_vals:
                  for g_val in g_vals: #this is causing the lists
                    list_x, list_u, list_Kmax, error, list_b, list_s ,der = model(days, pop, case, k_val, b_val, u0_val, sigma_val, Kmax_val, a_val, c_val, 'sigma', g_val, ID)
                    if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) < mse0:
                    #if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) > mse0:

                      mse0= mse(pop, list_x, error)
                      Dict['mse']= mse0
                      Dict['k_val']= k_val
                      Dict['Kmax0']= Kmax_val
                      Dict['b_val'] = b_val
                      Dict['case']= case
                      Dict['u0_val'] = u0_val
                      Dict['sigma_val']= sigma_val
                      Dict['a_val']=a_val
                      Dict['c_val']=c_val
                      Dict['g_val']= g_val
                    list_mse.append(mse(pop, list_x, error))
  return Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error
import statistics

def mse(x_true, x_pred,error):
  list_mse=[]
  #for i in range(len(x_true)):
    #for j in range(len(x_true[i])):
      #if (i) not in error:
        #if not( True in np.isnan(np.array(x_pred[i]))):
  list_mse.append(mean_absolute_error(x_true, x_pred))
  #list_mse.append(r2_score(x_true, x_pred))
  return statistics.mean(list_mse)

def get_error(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val):
  list_dict1=[]
  bad=0
  for i in range(len(scaled_pop)):
      #for j in range(len(list_days[i])):
        if (i) in group and (i) in response:
          try:

            D=gridsearch(days=scaled_days[i], pop= scaled_pop[i], model=run_model_fixed, k_vals=[ k_val],  b_vals=[b_val], cases=[ case],u0_vals=[u0_val], sigma_vals=[sigma_val], Kmax_vals=[Kmax_val], a_vals=[a_val], c_vals=[c_val], g_vals=[g_val])
            list_dict1.append(D)
          except:
            list_dict1.append({'mse':1000, 'Kmax0':0, 'k_val' : 0, 'b_val':0, 'case': 0,  'u0_val':0, 'sigma_val': 0 })
  good = 0
  list_error=[]
  for count in range(len(list_dict1)):
    if (list_dict1[count]['mse'])<0.1:# and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
      good +=1
    if list_dict1[count]['mse'] != 1000:
      list_error.append(list_dict1[count]['mse'])
    else:
      bad +=1


  return good, statistics.mean(list_error), len(list_dict1)-good-bad

def get_param(size, response):
  errors=[]
  Dict = {'mse':1000, 'k_val' : 0, 'b_val':0, 'case': 0, 'good':0 , 'u0_val': 0, 'K':0, 'a':0, 'c':0, 'g':0, 'sigma':0}
  for k in [ 0.2, 0.9, 2,  1, 5]:
    for b in [0.2,1, 2,10, 20]:
      for case in [ 'exp_K']:
        for u0 in [ 0.1]:
          for sigma in [ 0]:
            for K in [ 1, 2]:
              for a_val in [1]:
                for c_val in [   0.001, 0.01, 0.1, 0.2]: #C, this coresponds to d value in table 2
                  for g_val in [ 0.1, 0.5, 0.9]: #0.1, 0.5, 0.9
                    results =(get_error(size, response, k, b, case,u0, sigma, K, a_val, c_val, g_val))
                    #if results[0] >Dict['good'] or (results[0]== Dict['good'] and results[1]< Dict['mse']):
                    if  results[1]< Dict['mse']:

                      Dict['mse']= results[1]
                      Dict['k_val']= k
                      Dict['b_val'] = b
                      Dict['case']= case
                      Dict['good'] = results[0]
                      Dict['u0_val'] = u0
                      Dict['sigma']= sigma
                      Dict['K'] = K
                      Dict['a'] =a_val
                      Dict['c'] =c_val
                      Dict['g'] = g_val

  return Dict['mse'], Dict['k_val'], Dict['b_val'], Dict['case'], Dict['u0_val'], Dict['sigma'], Dict['K'], Dict['a'], Dict['c'], Dict['g']

def separate_by_size_predict_newdata4k_expK_all_m(tuple0): #free sigma for all dataset with MDA! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and tuple0 in Size1 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(9.602616255922462e-05, 2, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)



    group='Size1, Inc'
  elif len(Size1) > 0 and tuple0 in Size1 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(1.4924284769001023e-05, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)



  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0005581028152651971, 2, 0.2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)


    group='Size2, Inc'
  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0002474685799481297, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0008712237364711401, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)


    group='Size2, Inc'
  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00045665782397184303, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)


  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.003805638444789154, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size3, Inc'
  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00583392900567394, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)




  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_m2(tuple0): #free sigma for all dataset with MDA! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and tuple0 in Size1 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(4.9007374519872045e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)



    group='Size1, Inc'
  elif len(Size1) > 0 and tuple0 in Size1 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.46287476594794e-06, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)



  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.000208892569995844, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


    group='Size2, Inc'
  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(2.4680954119098068e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)


  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0008186625558637999, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size2, Inc'
  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0003065002240221485, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)


  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0020903478069908797, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)


    group='Size3, Inc'
  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0017453868791226252, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)





  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0


def separate_by_size_predict_newdata4k_expK_all_d(tuple0): #free sigma for all dataset with docetaxel! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and tuple0 in Size1 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.4973845374143e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

    group='Size1, Inc'
  elif len(Size1) > 0 and tuple0 in Size1 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(1.001913325901334e-05, 0.2, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.629359474403977e-05, 1, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size2, Inc'
  elif len(Size2) > 0 and tuple0 in Size2 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.240565937342348e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0005069629697468392, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

    group='Size2, Inc'
  elif len(Size3) > 0 and tuple0 in Size3 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00028086457826024576, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)


  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Inc:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.003322438544321211, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

    group='Size3, Inc'
  elif len(Size4) > 0 and tuple0 in Size4 and tuple0 in Dec:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0024268534223027454, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)



  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0


def separate_by_size(study, pop, arm):
  if arm == 'DOCETAXEL' or arm == 'docetaxel':
    k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_d(pop)
  elif arm == 'MPDL3280A':
    k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m(pop)
  else:
    k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0 = separate_by_size_predict_newdata4k_expK_all_m2(pop)


  return k0,b0,group, case0,  u0, sigma0, K0, a0, c0, g0

def PR_separate_by_size(study, arm, Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec):
  #C, Docetaxel is a chemotherapy medicine
  if arm == 'DOCETAXEL' or arm == 'docetaxel':
    k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_predict_newdata4k_expK_all_d(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
  #C, MPDL3280A_ also know as Atezolizumab, is immunotherapy
  elif arm == 'MPDL3280A':
    k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_predict_newdata4k_expK_all_m(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
  #C, others?
  else:
    k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_predict_newdata4k_expK_all_m2(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)

  return k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0

def PR_separate_by_size_predict_newdata4k_expK_all_d(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec): #free sigma for all dataset with docetaxel! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.4973845374143e-05, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

    group='Size1, Inc'
  elif len(Size1) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(1.001913325901334e-05, 0.2, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size2) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.629359474403977e-05, 1, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size2, Inc'
  elif len(Size2) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.240565937342348e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size3) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0005069629697468392, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9)

    group='Size2, Inc'
  elif len(Size3) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00028086457826024576, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)


  elif len(Size4) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.003322438544321211, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)

    group='Size3, Inc'
  elif len(Size4) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0024268534223027454, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)



  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0

def PR_separate_by_size_predict_newdata4k_expK_all_m(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec): #free sigma for all dataset with MDA! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(9.602616255922462e-05, 2, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.5)



    group='Size1, Inc'
  elif len(Size1) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(1.4924284769001023e-05, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.1)



  elif len(Size2) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0005581028152651971, 2, 0.2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)


    group='Size2, Inc'
  elif len(Size2) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0002474685799481297, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


  elif len(Size3) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0008712237364711401, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)


    group='Size2, Inc'
  elif len(Size3) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00045665782397184303, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)


  elif len(Size4) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.003805638444789154, 0.9, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size3, Inc'
  elif len(Size4) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00583392900567394, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)




  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0

def PR_separate_by_size_predict_newdata4k_expK_all_m2(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec): #free sigma for all dataset with MDA! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(4.9007374519872045e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)



    group='Size1, Inc'
  elif len(Size1) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.46287476594794e-06, 2, 10, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.9)



  elif len(Size2) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.000208892569995844, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1)


    group='Size2, Inc'
  elif len(Size2) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(2.4680954119098068e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.01, 0.1)


  elif len(Size3) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0008186625558637999, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)


    group='Size2, Inc'
  elif len(Size3) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0003065002240221485, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.5)


  elif len(Size4) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0020903478069908797, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)


    group='Size3, Inc'
  elif len(Size4) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0017453868791226252, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9)





  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0

# Extend variables to match the maximum length, filling in None for missing values
#Self made function
#Function to extend list to a given length with "Placeholder"
def extend_to_length(var, max_length):
  var_type = type(var)
  if isinstance(var, list) or  var_type  == 'gekko.gk_operators.GK_Value':
    return var + ["Placeholder"] * (max_length - len(var))
  elif isinstance(var, np.ndarray):
    placeholder_array = np.array(["Placeholder"] * (max_length - len(var)))
    return np.concatenate([var, placeholder_array])
  elif isinstance(var, (int, float, np.float64, str, np.int64)):
    return [var] * max_length
  else:
    print(f"Error: Unrecognized type {type(var)} for variable {var}")
    return [var] * max_length  # Default action

def print_variable_info(name, var):
    print(f"\n--- {name} ---")
    print(var)
    if isinstance(var, (list, np.ndarray)):
      print(f"Total Count: {len(var)}")
    elif isinstance(var, (int, float, np.float64)):
      print("This is a single numerical value, not a list.")
    print(f"Type: {type(var)}")

def PR_get_error(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val,scaled_pop, scaled_days, patient_list):

    list_dict1 = []
    bad = 0
    for i in range(len(scaled_pop)):
      # for j in range(len(list_days[i])):
      if (i) in group and (i) in response:
        try:

          D = gridsearch(days=scaled_days[i], pop=scaled_pop[i], model=run_model_fixed, k_vals=[k_val], b_vals=[b_val],
                         cases=[case], u0_vals=[u0_val], sigma_vals=[sigma_val], Kmax_vals=[Kmax_val], a_vals=[a_val],
                         c_vals=[c_val], g_vals=[g_val],ID=patient_list[i] )
          list_dict1.append(D)
        except:
          print(f"Except statement in, PR_get_error, for i: {i}")
          list_dict1.append({'mse': 1000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'sigma_val': 0})
    good = 0
    list_error = []
    for count in range(len(list_dict1)):
      if (list_dict1[count]['mse']) < 0.1:  # and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
        good += 1
      if list_dict1[count]['mse'] != 1000:
        list_error.append(list_dict1[count]['mse'])
      else:
        bad += 1

      print(f"{list_dict1}")
    return good, statistics.mean(list_error), len(list_dict1) - good - bad #C, this is the old one but mean give an error, so I removed it
    #return good, list_error, len(list_dict1) - good - bad

def PR_get_param(size, response, scaled_pop, scaled_days, patient_list):
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
                      results = (PR_get_error(size, response, k, b, case, u0, sigma, K, a_val, c_val, g_val, scaled_pop, scaled_days, patient_list))
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



def PR_get_error_Fixed(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val,scaled_pop, scaled_days, patient_list, Control):
  list_dict1 = []
  bad = 0
  for i in range(len(scaled_pop)):
    # for j in range(len(list_days[i])):
    if i in group and i in response:
      try:
        print("PR_get_error_Fixed block ")
        print(f"scaled_days: {scaled_days[i]}")
        print(f"scaled_pop: {scaled_pop[i]}")


        D = gridsearch_Fixed(days=scaled_days[i], pop=scaled_pop[i], model=run_model_fixed_fixed, k_vals=[k_val], b_vals=[b_val],
                             cases=[case], u0_vals=[u0_val], sigma_vals=[sigma_val], Kmax_vals=[Kmax_val], a_vals=[a_val],
                             c_vals=[c_val], g_vals=[g_val], ID=patient_list[i], Check=Control)
        list_dict1.append(D)
      except Exception as e:
        # This will catch any type of Exception
        print(f"An error occurred: {e}")
        print(f"Except statement in, PR_get_error_Fixed, for i: {i}")

        #Traceback
        print("Full exception traceback:")
        traceback.print_exc()

        list_dict1.append({'mse': 1000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'sigma_val': 0})
  good = 0
  list_error = []
  for count in range(len(list_dict1)):
    if (list_dict1[count]['mse']) < 0.1:  # and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
      good += 1
    if list_dict1[count]['mse'] != 1000:
      list_error.append(list_dict1[count]['mse'])
    else:
      bad +=1


      print(list_error)
  print(f"good: {good},statistics.mean{list_error}, list_dict1: {list_dict1}, bad: {bad} ")
  return good, statistics.mean(list_error), len(list_dict1) - good - bad


def PR_get_param_Fixed(size, response, scaled_pop, scaled_days, patient_list, Control):
  errors = []
  Dict = {'mse': 1000, 'k_val': 0, 'b_val': 0, 'case': 0, 'good': 0, 'u0_val': 0, 'K': 0, 'a': 0, 'c': 0, 'g': 0,
          'sigma': 0}
  for k in [0.2]:
    for b in [20]:
      for case in ['exp_K']:
        for u0 in [0.1]:
          for sigma in [0]:
            for K in [1]:
              for a_val in [1]:
                for c_val in [0.1]: #C, this coresponds to d value in table 2
                  for g_val in [0.9]:  # 0.1, 0.5, 0.9, (group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val,scaled_pop, scaled_days, patient_list, Control)
                    results = (PR_get_error_Fixed(group=size, response=response, k_val=k, b_val=b, case='exp_K', u0_val=u0, sigma_val=sigma, Kmax_val=K, a_val=a_val,
                                                  c_val=c_val, g_val=g_val , scaled_pop=scaled_pop, scaled_days=scaled_days, patient_list=patient_list, Control=Control))
                    # if results[0] >Dict['good'] or (results[0]== Dict['good'] and results[1]< Dict['mse']):
                    print(f"results[1]:{results[1]}, result: {results}, type:{type(results[1])}")
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


def gridsearch_Fixed(days, pop, model, k_vals,  b_vals, cases,u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, g_vals,ID, Check):
  list_mse=[]
  mse0=10000000
  #mse0=0
  print("Gridsearch_Fixed is getting called")
  Dict = {'mse':10000000, 'Kmax0':0, 'k_val' : 0, 'b_val':0, 'case': 0,  'u0_val':0, 'a_val':0, 'c_val':0, 'g_val':0 }
  for k_val in k_vals:
    Check = 0
    for Kmax_val in Kmax_vals:
      Check = 0
      for a_val in a_vals:
        Check = 0
        for c_val in c_vals:
          Check = 0
          for b_val in b_vals:
            Check = 0
            for case in cases:
              Check = 0
              for u0_val in u0_vals:
                Check = 0
                for sigma_val in sigma_vals:
                  Check = 0
                  for g_val in g_vals:
                    list_x, list_u, list_Kmax, error, list_b, list_s ,der = model(days, pop, case, k_vals, b_vals, u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, 'sigma', g_vals, ID, Check)
                    if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) < mse0:
                    #if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) > mse0:

                      mse0= mse(pop, list_x, error)
                      Dict['mse']= mse0
                      Dict['k_val']= k_val
                      Dict['Kmax0']= Kmax_val
                      Dict['b_val'] = b_val
                      Dict['case']= case
                      Dict['u0_val'] = u0_val
                      Dict['sigma_val']= sigma_val
                      Dict['a_val']=a_val
                      Dict['c_val']=c_val
                      Dict['g_val']= g_val
                    list_mse.append(mse(pop, list_x, error))
  return Dict

def run_model_fixed_fixed(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val,free, g_val, ID,Check):

  #C, for some reason the variables are list so let fix that.I found it, I on accident sent the whole list, instead of the
  #element of the list. Going to fix that later, see for difference, gridsearch and gridsearch_fixed
  # Extracting the first element from each list and overwriting the original variables
  k_val = k_val[0]
  b_val = b_val[0]
  u0_val = u0_val[0]
  sigma_val = sigma_val[0]
  Kmax0 = Kmax0[0]
  a_val = a_val[0]
  c_val = c_val[0]
  g_val = g_val[0]

  ##Debugging
  print("Run_model_fixed_fixed is runn")
  # Printing the values and types in days
  print("\n--- days ---")
  print(days)
  print(f"Total Count: {len(days) if isinstance(days, list) else 'Not a list'}")
  print(f"Type: {type(days)}")

  # Printing the values and types in population
  print("\n--- population ---")
  print(population)
  print(f"Total Count: {len(population) if isinstance(population, list) else 'Not a list'}")
  print(f"Type: {type(population)}")

  # Printing the values and types in case
  print("\n--- case ---")
  print(case)
  print(f"Type: {type(case)}")

  # Printing the values and types in k_val
  print("\n--- k_val ---")
  print(k_val)
  print(f"Type: {type(k_val)}")

  # Printing the values and types in b_val
  print("\n--- b_val ---")
  print(b_val)
  print(f"Type: {type(b_val)}")

  # Printing the values and types in u0_val
  print("\n--- u0_val ---")
  print(u0_val)
  print(f"Type: {type(u0_val)}")

  # Printing the values and types in sigma_val
  print("\n--- sigma_val ---")
  print(sigma_val)
  print(f"Type: {type(sigma_val)}")

  # Printing the values and types in Kmax0
  print("\n--- Kmax0 ---")
  print(Kmax0)
  print(f"Type: {type(Kmax0)}")

  # Printing the values and types in a_val
  print("\n--- a_val ---")
  print(a_val)
  print(f"Type: {type(a_val)}")

  # Printing the values and types in c_val
  print("\n--- c_val ---")
  print(c_val)
  print(f"Type: {type(c_val)}")

  # Printing the values and types in free
  print("\n--- free ---")
  print(free)
  print(f"Type: {type(free)}")

  # Printing the values and types in g_val
  print("\n--- g_val ---")
  print(g_val)
  print(f"Type: {type(g_val)}")

  # Printing the values and types in ID
  print("\n--- ID ---")
  print(ID)
  print(f"Type: {type(ID)}")

  # Printing the values and types in Check
  print("\n--- Check ---")
  print(Check)
  print(f"Type: {type(Check)}")
  print("END OF RUN MODEL DEBUGGING")
  #g_val = 0.5, this was in the defautl setting, removed to the body
  list_x =[]
  list_u =[]
  list_Kmax =[]
  list_b =[]
  error=[]
  der=[]
  list_s=[]
  #try:
  m = GEKKO(remote=False)
  m.time= days
  x_data= population
  x = m.CV(value=x_data, lb=0); x.FSTATUS = 1 # fit to measurement
  x.SPLO = 0
  if free =='sigma':
    #sigma= m.Param(0)
    sigma = m.FV(value=0.01, lb= 0, ub=0.1); sigma.STATUS=1
    #sigma = m.FV(value=0.01, lb= 0, ub=1); sigma.STATUS=1

    k = m.Param(k_val)
  elif free== 'k':
    sigma= m.Param(0)
    k = m.Param(k_val)

    #k = m.FV(value=0.1, lb= 0, ub=10); k.STATUS=1
  d = m.Param(c_val)
  b = m.Param(b_val)
  #g_val = m.FV(value=0.5, lb= 0); g_val.STATUS=1


  r = m.FV(value=0.4, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001
  #r = m.FV(value=c_val, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  #r = m.FV(value=0.01, lb=0.00001, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  step = [0 if z<0 else 1 for z in m.time]
  m_param = m.Param(1)
  u = m.Var(value=u0_val, lb=0)#, ub=1)
  m.free(u)
  a = m.Param(a_val)
  c= m.Param(c_val)
  Kmax= m.Param(Kmax0)
  if case == 'case3':
    m.Equations([x.dt()==  (x)*(r*(1-u)*(1-x/Kmax)-m_param/(k+b*u)-d), u.dt()==sigma*(b*m_param/((b*u+k)**2)-r*(1-x/(Kmax)))])
  elif case == 'case0':
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt()== sigma*(m_param*b/((k+b*u)**2))])
  elif case == 'case4':
    m.Equations([x.dt()==  x*(r*(1-u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'case5':
    m.Equations([x.dt()==  x*(r*(1+u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_r':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(1-x/(Kmax))*(m.exp(-g_val*u))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K':
    # print(f"right statement, exp_K block_Fixed_fixed")
    # #u unbounded for this one
    # print("\n------Debuging equation------")
    # print(f"\nEntering case 'exp_K'")
    #
    # # Printing the variables and their types before they are used
    # print(f"g_val: {g_val}, Type: {type(g_val)}")
    # print(f"u: {u}, Type: {type(u)}")
    # print(f"r: {r}, Type: {type(r)}")
    # print(f"x: {x}, Type: {type(x)}")
    # print(f"Kmax: {Kmax}, Type: {type(Kmax)}")
    # print(f"m_param: {m_param}, Type: {type(m_param)}")
    # print(f"k: {k}, Type: {type(k)}")
    # print(f"b: {b}, Type: {type(b)}")
    # print(f"d: {d}, Type: {type(d)}")
    #
    # # Printing the intermediate values and their types
    # inner_exp_value = -g_val * u
    # print(f"Inner exp value: {inner_exp_value}, Type: {type(inner_exp_value)}")
    #
    # exp_value = m.exp(inner_exp_value)
    # print(f"Exp value: {exp_value}, Type: {type(exp_value)}")
    #
    # kmax_exp_value = Kmax * exp_value
    # print(f"Kmax * exp value: {kmax_exp_value}, Type: {type(kmax_exp_value)}")
    #
    # x_over_kmax_exp_value = x / kmax_exp_value
    # print(f"x / (Kmax * exp) value: {x_over_kmax_exp_value}, Type: {type(x_over_kmax_exp_value)}")
    #
    # r_term_value = r * (1 - x_over_kmax_exp_value)
    # print(f"r * (1 - x / (Kmax * exp)) value: {r_term_value}, Type: {type(r_term_value)}")
    #
    # m_param_over_k_b_u_value = m_param / (k + b * u)
    # print(f"m_param / (k + b * u) value: {m_param_over_k_b_u_value}, Type: {type(m_param_over_k_b_u_value)}")
    #
    # final_x_dt_value = r_term_value - m_param_over_k_b_u_value - d
    # print(f"Final x.dt() value: {final_x_dt_value}, Type: {type(final_x_dt_value)}")
    #
    # u_dt_part1_value  = (-g_val * r * x * m.exp(g_val * u)) / Kmax
    # print(f"u.dt part value before addition: {u_dt_part1_value}, Type: {type(u_dt_part1_value)}")
    #
    # u_dt_part2_value = (b * m_param) / ((k + b * u) ** 2)
    # print(f"u.dt part2 value: {u_dt_part2_value}, Type: {type(u_dt_part2_value)}")
    #
    # u_dt_value = sigma * (u_dt_part1_value + u_dt_part2_value)  # Added sigma multiplication here
    # print(f"u.dt() value: {u_dt_value}, Type: {type(u_dt_value)}")
    #
    # # Original equations
    # m.Equations([x.dt() == final_x_dt_value,
    #              u.dt() == u_dt_value])
    #
    # print("Equations set.")
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K_benefit':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  elif case == 'exp_both':
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(m.exp(-g_val*u))*(1-x*(m.exp(g_val*u))/(Kmax))+(b*m_param)/((b*u+k)**2)-g_val*r*x/(Kmax))])
  elif case == 'exp':
    m.Equations([x.dt()==  x*(r*(1-u)*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-r*(1-(x*m.exp(g_val*u))/Kmax)-(g_val*r*x*(1-u)*m.exp(g_val*u))/Kmax+(b*m)/(b*u+k)**2)])
  elif case == 'exp_K_neg':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == -sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  m.options.IMODE = 5  # dynamic estimation
  m.options.NODES = 5   # collocation nodes
  m.options.EV_TYPE = 2 # linear error (2 for squared)
  #m.options.MAX_ITER=15000

  m.solve(disp=False, debug=False)    # do not display solver output
  list_x =x.value
  list_u= u.value
  der = Kmax.value[0]
  list_Kmax= r.value[0]
  list_b = k.value[0]
  list_s = sigma.value[0]
  '''except:
  list_x =0
  list_u= 0
  der = 0
  list_Kmax= 0
  list_b = 0
  list_s = 0'''

  ##Getting stuff
  formatted_time = datetime.now().strftime("%d_%H_%M_%S")

  #Setting path
  dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"
  sys.path.insert(0, dataset_path)

  target_dir = os.path.join(dataset_path, "Edited", "Model_params", "Run_Singles_Run_model_fixed_fixed", "Index_Kept",f"{ID}")
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  #Check statement
  Check = Check + 1

  # Using the function to print the information of variables
  print(f"\n--------- RUN_MODEL_FIXED_FIXED_after_equation check-------------------")
  print(f"PatientID: {ID}")
  print(f"Time: {formatted_time}")
  print(f"Check: {Check}")

  print_variable_info('list_x', list_x)
  print_variable_info('list_u', list_u)
  print_variable_info('list_Kmax', list_Kmax)
  print_variable_info('error', error)
  print_variable_info('list_b', list_b)
  print_variable_info('list_s', list_s)
  print_variable_info('der', der)

  variables = [list_x, list_u, list_Kmax, error, list_b, list_s, der]
  lengths = []

  for var in variables:
    var_type = type(var)
    if isinstance(var, list) or var_type  == 'gekko.gk_operators.GK_Value':
      lengths.append(len(var))
    elif isinstance(var, (int, float)):
      # We won't add this to the lengths list as it's a single value
      pass
    else:
      print(f"Warning: Unhandled type {type(var)} for variable {var}")

  # Determining max length
  if lengths:  # Check if lengths is not empty to avoid ValueError
    max_length = max(lengths)
  else:
    print("Error: No iterable variables to determine max length.")
    max_length = 0

  # Using your extend_to_length function to make all lists the same length, bcs dont know if it is all different lengths
  extended_list_x = extend_to_length(list_x, max_length)
  extended_list_u = extend_to_length(list_u, max_length)
  extended_list_Kmax = extend_to_length(list_Kmax, max_length)
  extended_error = extend_to_length(error, max_length)
  extended_list_b = extend_to_length(list_b, max_length)
  extended_list_s = extend_to_length(list_s, max_length)
  extended_der = extend_to_length(der, max_length)
  extended_ID = extend_to_length(ID,max_length)

  # Save to CSV
  filename = os.path.join(target_dir, f"{formatted_time}" + f"_{Check}" + ".csv")
  with builtins.open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ID","list_x", "list_u", "list_Kmax", "error", "list_b", "list_s", "der"])
    writer.writerows(
      zip(extended_ID, extended_list_x, extended_list_u, extended_list_Kmax, extended_error, extended_list_b, extended_list_s,
          extended_der))

  return list_x, list_u, list_Kmax, error, list_b, list_s, der





###THE VERSION OF GET_PARAM, THAT DOESNT SAVE

def PR_get_param_unsave(size, response, scaled_pop, scaled_days):
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
                      results = (PR_get_error_unsaved(size, response, k, b, case, u0, sigma, K, a_val, c_val, g_val, scaled_pop,
                                              scaled_days))
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
      Dict['c'], Dict['g'] #C, what is function of \

def PR_get_error_unsaved(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val, scaled_pop,
                 scaled_days):
  list_dict1 = []
  bad = 0
  for i in range(len(scaled_pop)):
    # for j in range(len(list_days[i])):
    if (i) in group and (i) in response:
      try:

        D = gridsearch_unsaved(days=scaled_days[i], pop=scaled_pop[i], model=run_model_fixed_unsaved, k_vals=[k_val], b_vals=[b_val],
                       cases=[case], u0_vals=[u0_val], sigma_vals=[sigma_val], Kmax_vals=[Kmax_val], a_vals=[a_val],
                       c_vals=[c_val], g_vals=[g_val])
        list_dict1.append(D)
      except:
        print(f"Except statement in, PR_get_error, for i: {i}")
        list_dict1.append({'mse': 1000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'sigma_val': 0})
  good = 0
  list_error = []
  for count in range(len(list_dict1)):
    if (list_dict1[count]['mse']) < 0.1:  # and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
      good += 1
    if list_dict1[count]['mse'] != 1000:
      list_error.append(list_dict1[count]['mse'])
    else:
      bad += 1


    print()
  return good, statistics.mean(list_error), len(
    list_dict1) - good - bad  # C, this is the old one but mean give an error, so I removed it
  # return good, list_error, len(list_dict1) - good - bad

def gridsearch_unsaved(days, pop, model, k_vals,  b_vals, cases,u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, g_vals):
  list_mse=[]
  mse0=10000000
  #mse0=0

  Dict = {'mse':10000000, 'Kmax0':0, 'k_val' : 0, 'b_val':0, 'case': 0,  'u0_val':0, 'a_val':0, 'c_val':0, 'g_val':0 }
  for k_val in k_vals:
    for Kmax_val in Kmax_vals:
      for a_val in a_vals:
        for c_val in c_vals:
          for b_val in b_vals:
            for case in cases:
              for u0_val in u0_vals:
                for sigma_val in sigma_vals:
                  for g_val in g_vals:
                    list_x, list_u, list_Kmax, error, list_b, list_s ,der = model(days, pop, case, k_val, b_val, u0_val, sigma_val, Kmax_val, a_val, c_val, 'sigma', g_val)
                    #print(f"pop: {pop}")
                    #print(f"list_x: {list_x}")
                    #print(f"error: {error}")#
                    #
                    #print(f"mse0: {mse0}")
                    if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) < mse0:
                    #if mse(pop, list_x, error)==float(mse(pop, list_x, error)) and mse(pop, list_x, error) > mse0:

                      mse0= mse(pop, list_x, error)
                      Dict['mse']= mse0
                      Dict['k_val']= k_val
                      Dict['Kmax0']= Kmax_val
                      Dict['b_val'] = b_val
                      Dict['case']= case
                      Dict['u0_val'] = u0_val
                      Dict['sigma_val']= sigma_val
                      Dict['a_val']=a_val
                      Dict['c_val']=c_val
                      Dict['g_val']= g_val
                    list_mse.append(mse(pop, list_x, error))
  return Dict

def run_model_fixed_unsaved(days, population, case, k_val, b_val, u0_val, sigma_val, Kmax0, a_val, c_val, free, g_val=0.5):
  #print("It is running")
  list_x =[]
  list_u =[]
  list_Kmax =[]
  list_b =[]
  error=[]
  der=[]
  list_s=[]
  #try:
  m = GEKKO(remote=False)
  m.time= days
  x_data= population
  x = m.CV(value=x_data, lb=0); x.FSTATUS = 1 # fit to measurement
  x.SPLO = 0
  if free =='sigma':
    #sigma= m.Param(0)
    sigma = m.FV(value=0.01, lb= 0, ub=0.1); sigma.STATUS=1
    #sigma = m.FV(value=0.01, lb= 0, ub=1); sigma.STATUS=1

    k = m.Param(k_val)
  elif free== 'k':
    sigma= m.Param(0)
    k = m.Param(k_val)

    #k = m.FV(value=0.1, lb= 0, ub=10); k.STATUS=1
  d = m.Param(c_val)
  b = m.Param(b_val)
  #g_val = m.FV(value=0.5, lb= 0); g_val.STATUS=1


  r = m.FV(value=0.4, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001
  #r = m.FV(value=c_val, lb=c_val, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  #r = m.FV(value=0.01, lb=0.00001, ub=1); r.STATUS=1 #, ub=a_val  #estaba en 0.01 y lb= 0.00001

  step = [0 if z<0 else 1 for z in m.time]
  m_param = m.Param(1)
  u = m.Var(value=u0_val, lb=0)#, ub=1)
  m.free(u)
  a = m.Param(a_val)
  c= m.Param(c_val)
  Kmax= m.Param(Kmax0)
  if case == 'case3':
    m.Equations([x.dt()==  (x)*(r*(1-u)*(1-x/Kmax)-m_param/(k+b*u)-d), u.dt()==sigma*(b*m_param/((b*u+k)**2)-r*(1-x/(Kmax)))])
  elif case == 'case0':
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt()== sigma*(m_param*b/((k+b*u)**2))])
  elif case == 'case4':
    m.Equations([x.dt()==  x*(r*(1-u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'case5':
    m.Equations([x.dt()==  x*(r*(1+u**2)*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(2*u*r*(1-x/(Kmax))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_r':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(1-x/(Kmax))*(m.exp(-g_val*u))+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])
  elif case == 'exp_K_benefit':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*((g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  elif case == 'exp_both':
    m.Equations([x.dt()==  x*(r*(m.exp(-g_val*u))*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-g_val*r*(m.exp(-g_val*u))*(1-x*(m.exp(g_val*u))/(Kmax))+(b*m_param)/((b*u+k)**2)-g_val*r*x/(Kmax))])
  elif case == 'exp':
    m.Equations([x.dt()==  x*(r*(1-u)*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == sigma*(-r*(1-(x*m.exp(g_val*u))/Kmax)-(g_val*r*x*(1-u)*m.exp(g_val*u))/Kmax+(b*m)/(b*u+k)**2)])
  elif case == 'exp_K_neg':
    #u unbounded for this one
    m.Equations([x.dt()==  x*(r*(1-x/(Kmax*(m.exp(-g_val*u))))-m_param/(k+b*u)-d), u.dt() == -sigma*((-g_val*r*x*(m.exp(g_val*u)))/(Kmax)+(b*m_param)/(b*u+k)**2)])

  m.options.IMODE = 5  # dynamic estimation
  m.options.NODES = 5   # collocation nodes
  m.options.EV_TYPE = 2 # linear error (2 for squared)
  #m.options.MAX_ITER=15000

  m.solve(disp=False, debug=False)    # do not display solver output
  list_x =x.value
  list_u= u.value
  der = Kmax.value[0]
  list_Kmax= r.value[0]
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

def PR_get_param_unsaved_gpu(size, response, scaled_pop, scaled_days):
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
                      results = (PR_get_error_unsaved_gpu(size, response, k, b, case, u0, sigma, K, a_val, c_val, g_val, scaled_pop,
                                              scaled_days))
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
      Dict['c'], Dict['g'] #C, what is function of \


def PR_get_error_unsaved_gpu(group, response, k_val, b_val, case, u0_val, sigma_val, Kmax_val, a_val, c_val, g_val, scaled_pop,
                 scaled_days):
  list_dict1 = []
  bad = 0
  for i in range(len(scaled_pop)):
    # for j in range(len(list_days[i])):
    if (i) in group and (i) in response:
      try:
        # Convert the arrays to CuPy arrays here
        scaled_pop_gpu = cp.array(scaled_pop[i])
        scaled_days_gpu = cp.array(scaled_days[i])

        D = gridsearch_unsaved(days=scaled_days_gpu, pop=scaled_pop_gpu, model=run_model_fixed_unsaved, k_vals=[k_val], b_vals=[b_val],
                       cases=[case], u0_vals=[u0_val], sigma_vals=[sigma_val], Kmax_vals=[Kmax_val], a_vals=[a_val],
                       c_vals=[c_val], g_vals=[g_val])
        list_dict1.append(D)

      except Exception as e:
        print("An error occurred:", e)
        print("Traceback:")
        traceback.print_exc()  # This line prints the entire traceback
        print(f"Except statement in, PR_get_error_GPU, for i: {i}")
        list_dict1.append({'mse': 1000, 'Kmax0': 0, 'k_val': 0, 'b_val': 0, 'case': 0, 'u0_val': 0, 'sigma_val': 0})
  good = 0
  list_error = []
  for count in range(len(list_dict1)):
    if (list_dict1[count]['mse']) < 0.1:  # and (list_dict[count]['k_val'])==0.1 and (list_dict[count]['b_val'])==0.02:
      good += 1
    if list_dict1[count]['mse'] != 1000:
      list_error.append(list_dict1[count]['mse'])
    else:
      bad += 1



  return good, statistics.mean(list_error), len(list_dict1) - good - bad  # C, this is the old one but mean give an error, so I removed it
  # return good, list_error, len(list_dict1) - good - bad



def PR_separate_by_size_chemo_vs_immuno(study, arm, Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec):
  #C, Docetaxel is a chemotherapy medicine
  if arm == 'DOCETAXEL' or arm == 'docetaxel':
    k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_predict_newdata4k_expK_all_d_chemo(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)
  #C, everyhing immuno
  else:
    k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0 = PR_separate_by_size_predict_newdata4k_expK_all_m2_immuno(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec)

  return k0, b0, group, case0, u0, sigma0, K0, a0, c0, g0

def PR_separate_by_size_predict_newdata4k_expK_all_d_chemo(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec): #free sigma for all dataset with docetaxel! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.063877969429763e-05, 2, 2, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.5) #Changed params

    group='Size1, Inc'
  elif len(Size1) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.861986421844076e-06, 2, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.001, 0.5) #Changed params


  elif len(Size2) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(8.629359474403977e-05, 1, 2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9) #Changed params


    group='Size2, Inc'
  elif len(Size2) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(6.240565937342348e-05, 0.9, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.1) #Chamged params


  elif len(Size3) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0007234121005921019, 1, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.2, 0.9) #Changed params

    group='Size3, Inc' #For some reason this was size 2
  elif len(Size3) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00028153314534646944, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5)#Changed params


  elif len(Size4) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.003651193853480923, 1, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.001, 0.9)#Changed params

    group='Size4, Inc' #For some reason this was size 3
  elif len(Size4) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0025708924560113172, 0.9, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.1, 0.9) #Changed params



  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0

def PR_separate_by_size_predict_newdata4k_expK_all_m2_immuno(Size1, Size2, Size3, Size4, Up, Down, Fluctuate, Evolution, Inc, Dec): #free sigma for all dataset with MDA! 4 size groups
  k0=0
  b0=0
  case0 ='c'
  group='s'
  K0=2
  r =0
  sigma=0
  u0=0
  a0=0
  c0=0
  K0=0
  g0=0
  if len(Size1) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00010029458316055608, 0.9, 10, 'exp_K', 0.1, 0.01, 2, 0.7, 0.1, 0.5) #Changed params



    group='Size1, Inc'
  elif len(Size1) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(1.352026405124126e-05, 2, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.1, 0.1) #Changed params



  elif len(Size2) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00036980020198966966, 5, 0.2, 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.9) #Changed params


    group='Size2, Inc'
  elif len(Size2) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0001375282587590039, 1, 1, 'exp_K', 0.1, 0.01, 1, 0.7, 0.001, 0.1) #Changed params


  elif len(Size3) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0008634362741894515, 0.9, 1 , 'exp_K', 0.1, 0.01, 2, 0.7, 0.001, 0.9) #Changed params


    group='Size3, Inc' #For some reason this was size 2
  elif len(Size3) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.00041855591024892266, 0.2, 20 , 'exp_K', 0.1, 0.01, 1, 0.7, 0.2, 0.5) #Changed params


  elif len(Size4) > 0 and len(Inc) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.0031425379876819847, 1, 1, 'exp_K', 0.1, 0.01, 2, 0.7, 0.1, 0.9) #Changed params


    group='Size4, Inc' #For some reason this was size 3
  elif len(Size4) > 0 and len(Dec) > 0:
    error, k0, b0, case0, u0, sigma, K0, a0, c0,g0 =(0.004157316515650537, 0.2, 20, 'exp_K', 0.1, 0.01, 1, 0.7, 0.001, 0.5) #Changed params




  return k0,b0, group, case0,  u0, sigma, K0, a0, c0, g0



#Just added sim for firstDim:
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

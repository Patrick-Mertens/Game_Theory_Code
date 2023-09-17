#This contains the self written functions from the nsclc_paper.ipyn

from math import pi
import statistics
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

  def Detect_Trend_Of_Data(vector): #C, this one gives an error

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


#function to
def gridsearch(days, pop, model, k_vals,  b_vals, cases,u0_vals, sigma_vals, Kmax_vals, a_vals, c_vals, g_vals):
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
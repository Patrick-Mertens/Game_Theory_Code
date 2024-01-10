#C, this script attempts to replicate the parameter of table 2


#Librareis part 1
import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statistics

#Part 2
import warnings
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import Utils as utils
import FitFunctions as ff
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Normalizer
import matplotlib as mpl
from matplotlib.lines import Line2D
import sys
import builtins
import csv

#Importing the local functions
from VG_Functions_2 import *



#####Initial conditions#############
#Currently only doing Size1, increase, however, code will be written in a way that it can easily be extneded
sizes = ["Size1"]
movements = ["Increase"]  # Add more movements if needed
#This file was creeated in R script
path_to_csv = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\PatientID\df_Size1_and_Increase.csv"

#Set dataset_path, on folder back where the pickle files are located
base_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited"  # Use a raw string for the path
sys.path.insert(0,base_path)


##Options for the for loop##
#Activate the print function in the for loop
Debugging = True
#Saving intermediary result
Saving = True
#Doing the param determination of the for loop
param = True

##

##Study settings##
studies = [f"study{i}" for i in range(1, 4)]
functions = ['Exponential']
splits = [True, True, False, True] #C, removed the last one
trends = ['Up', 'Down', 'Fluctuate', 'Evolution']
noPars = [3, 3, 3, 4, 3, 4]


#########################################

#For loop
for Size in sizes:
    #Selecting size

    for Movement in movements:
        #Selecting the movement

        # Combine dataframes from study 1 to 4
        dfs = []
        for study in studies:
            df_path = os.path.join(base_path, "SpiderProject_KatherLab", 'Study_' + study + '_1.xlsx')
            #dfs.append(pd.read_csv(df_path))

            #C, old code stuff
            sind = studies.index(study)#Previously studyname
            sp = splits[sind]
            if Debugging == True:
                #Debugging
                print(f"Study path: {df_path}")
                print(f"Study: {study}")
                print(f"Data Frame: {dfs}")



        combined_df = pd.concat(dfs, axis=0)

        # If the conditions are met, read the csv and filter the combined dataframe
        if Size == "Size1" and Movement == "Increase":
            patient_ids_df = pd.read_csv(path_to_csv)
            combined_df = combined_df[combined_df['PatientID'].isin(patient_ids_df['PatientID'])]

        # Creating  the dataframe into subsets
        chemotherapy = combined_df[combined_df['arm'].isin(['DOCETAXEL', 'docetaxel'])]
        immunotherapy = combined_df[combined_df['arm'] == 'MPDL3280A']
        #if it is not in chemotherapy and immunotherapy
        unknown_group = combined_df[~combined_df['arm'].isin(['DOCETAXEL', 'docetaxel', 'MPDL3280A'])]

        if Debugging == True:
            # Debugging
            # Print length of each subset
            print(f"Length of chemotherapy group: {len(chemotherapy)}")
            print(f"Length of immunotherapy group: {len(immunotherapy)}")
            print(f"Length of unknown group: {len(unknown_group)}")


        # Check folder existence and save
        target_dir = os.path.join(base_path, "Param", f"{Size}", f"{Movement}")
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        if Saving == True:
            # Check folder existence
            target_dir = os.path.join(base_path, "Param", f"{Size}", f"{Movement}")
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            #Saving the files
            chemotherapy.to_csv(os.path.join(target_dir, "chemotherapy.csv"), index=False)
            immunotherapy.to_csv(os.path.join(target_dir, "immunotherapy.csv"), index=False)
            unknown_group.to_csv(os.path.join(target_dir, "unknown_group.csv"), index=False)


        if param == True:
            # Gridsearch and get results
            # Assuming you have methods to convert these dataframes into suitable format for your gridsearch function
            chemo_results = gridsearch(chemotherapy)
            immuno_results = gridsearch(immunotherapy)
            unknown_results = gridsearch(unknown_group)

        # You can save these results or process them further based on your requirements

import os
import pickle
import numpy as np
import pandas as pd

dataset_path = "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab"
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'Gompertz_orginial',
             'GeneralGompertz']
functions = ['Exponential']
seeds = range(42, 44)
iterations = range(1, 3)
studies = ["1", "2", "3", '4']
trends = ["Up", "Down", "Evolution", "Fluctuate"]

for functionToFit in functions:
    print(f"\nFit Function: {functionToFit}")
    for seed in seeds:
        print(f"\nSeed: {seed}")
        for trend in trends:
            print(f"\nTrend: {trend}")

            all_arms = set()  # Set to gather all unique arms
            rSquared_values = {}  # Dictionary to store r-squared values

            for run in iterations:
                for study in studies:
                    file_path = os.path.join(dataset_path, functionToFit, 'Runner_scriptstudy1to4', f"seed_{seed}",
                                             f"for_loop_itteration_{run}", 'Combo_formula', f"{study}.pkl")
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as file:
                            result_dict = pickle.load(file)
                        for arm in result_dict.keys():
                            if result_dict[arm][trend]['rSquare']:
                                rSquared = np.nanmean(result_dict[arm][trend]['rSquare'])
                                rSquared_values.setdefault(arm, {})[f"Iteration {run}"] = np.around(rSquared, 6)
                            all_arms.add(arm)

            # Convert the rSquared_values dictionary to a DataFrame
            df = pd.DataFrame.from_dict(rSquared_values, orient='index')

            # Display the DataFrame
            print(df)

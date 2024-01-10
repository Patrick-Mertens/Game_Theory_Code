import os
import pickle
import numpy as np
import pandas as pd

dataset_path = "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab"
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'Gompertz_orginial',
             'GeneralGompertz']
seeds = range(42, 46)
iterations = range(1, 11)
studies = ["1", "2", "3", '4', '5']
trends = ["Up", "Down", "Evolution", "Fluctuate"]

for functionToFit in functions:
    print(f"\nFit Function: {functionToFit}")
    for seed in seeds:
        print(f"\nSeed: {seed}")
        for trend in trends:
            print(f"\nTrend: {trend}")
            # Initialize a dataframe to store the R-squared values
            df = pd.DataFrame(index=studies, columns=[f"Iteration {i}" for i in iterations])
            for run in iterations:
                for study in studies:
                    file_path = os.path.join(dataset_path, functionToFit, 'Runner_script', f"seed_{seed}",
                                             f"for_loop_itteration_{run}", 'Combo_formula', f"{study}.pkl")
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as file:
                            result_dict = pickle.load(file)
                        for arm in sorted(result_dict.keys()):
                            rSquared = np.nanmean(result_dict[arm][trend]['rSquare']) if result_dict[arm][trend][
                                'rSquare'] else np.nan
                            df.at[arm, f"Iteration {run}"] = np.around(rSquared, 6)

            # Display the dataframe
            print(df)

# import os
# import pickle
# import numpy as np
#
#
#
# ###
# dataset_path = "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab"
# functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz','Gompertz_orginial', 'GeneralGompertz'] # List all the fit functions you have
# seeds = range(42, 46)
# iterations = range(1, 11)
# studies = ["1", "2", "3",'4','5']  # List all the study numbers you have
#
# for functionToFit in functions:
#     for seed in seeds:
#         rSquared_values_per_iteration = {}
#         for run in iterations:
#             path = os.path.join(dataset_path, functionToFit, 'Runner_script', f"seed_{seed}",
#                                 f"for_loop_itteration_{run}", 'Combo_formula')
#             rSquared_values = {}
#             for study in studies:
#                 file_path = os.path.join(path, f"{study}.pkl")
#                 if os.path.exists(file_path):
#                     with open(file_path, "rb") as file:
#                         result_dict = pickle.load(file)
#                     arms = sorted(result_dict.keys())
#                     for arm in arms:
#                         for trend in ["Up","Down","Evolution", "Fluctuate"]:  # Add any other trends you have
#                             key = f"{arm}_{trend}"
#                             rSquared = np.nanmean(result_dict[arm][trend]['rSquare'])
#                             rSquared_values[key] = np.around(rSquared, 6)
#             rSquared_values_per_iteration[run] = rSquared_values
#
#         # Check if Rsquared values are consistent across iterations
#         base_iteration = rSquared_values_per_iteration[1]  # Assuming first iteration as the base
#         for iteration, values in rSquared_values_per_iteration.items():
#             if values != base_iteration:
#                 print(f"Inconsistency found in {functionToFit}, seed {seed}, iteration {iteration}")


#####

import os
import pickle
import numpy as np

dataset_path = "C:/Users/Shade/Desktop/Master/Project Game Theory Code/Downloaded file/Edited/SpiderProject_KatherLab"
functions = ['Exponential', 'Logistic', 'ClassicBertalanffy', 'GeneralBertalanffy', 'Gompertz', 'Gompertz_orginial',
             'GeneralGompertz']
seeds = range(42, 46)
iterations = range(1, 11)
studies = ["1", "2", "3", '4', '5']
trends = ["Up", "Down", "Evolution", "Fluctuate"]

for functionToFit in sorted(functions):
    print(f"\nChecking {functionToFit}")
    for seed in sorted(seeds):
        print(f"\nSeed: {seed}")
        rSquared_values_per_iteration = {trend: {arm: {} for arm in sorted(studies)} for trend in sorted(trends)}

        for run in sorted(iterations):
            path = os.path.join(dataset_path, functionToFit, 'Runner_script', f"seed_{seed}",
                                f"for_loop_itteration_{run}", 'Combo_formula')

            for study in sorted(studies):
                file_path = os.path.join(path, f"{study}.pkl")
                if os.path.exists(file_path):
                    with open(file_path, "rb") as file:
                        result_dict = pickle.load(file)
                    arms = sorted(result_dict.keys())
                    for arm in sorted(arms):
                        for trend in sorted(trends):
                            key = f"{arm}_{trend}"
                            if result_dict[arm][trend]['rSquare']:
                                rSquared = np.nanmean(result_dict[arm][trend]['rSquare'])
                                rSquared_values_per_iteration[trend][arm][run] = np.around(rSquared, 6)
                            else:
                                rSquared_values_per_iteration[trend][arm][run] = np.nan
                            if len(result_dict[arm][trend]['data']) > 1:
                                print(
                                    f"Iteration {run}, Arm {arm}, Trend {trend}: {result_dict[arm][trend]['data'][1]}")

        # Check if Rsquared values are consistent across trends and arms
        for trend in sorted(trends):
            print(f"\nTrend: {trend}")
            for arm in sorted(studies):
                base_iteration = rSquared_values_per_iteration[trend][arm][1]  # Assuming first iteration as the base
                inconsistent_iterations = []
                for iteration in sorted(iterations):
                    if iteration != 1:  # Skip the base iteration
                        current_value = rSquared_values_per_iteration[trend][arm][iteration]
                        if current_value != base_iteration:
                            inconsistent_iterations.append((iteration, current_value))
                if inconsistent_iterations:
                    print(f"Inconsistencies in arm {arm}: {inconsistent_iterations}")
                else:
                    print(f"Arm {arm} is consistent across all iterations.")

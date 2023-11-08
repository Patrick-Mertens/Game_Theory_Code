import os
import pandas as pd
import glob

# The path format with placeholders for the sigma and the variable part (1, 2, or 3)
path_format = "/home/bob/project_game_theory_linux/SpiderProject_KatherLab/Results_nsclc_paper_3/PR_exp_K_sigma_0/{variable}/Sigma_0/Exponential/4_November"
sigmas = ['0']  # Possible sigma values
variables = ['1', '2', '3','4']  # The variable part of the path that changes

# Loop over the possible sigma and variable values to find the correct path
for sigma in sigmas:
    for variable in variables:
        # Generate the path
        path = path_format.format(sigma=sigma, variable=variable)
        # List all CSV files in this directory
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                print('test')
                # Check the first value in the 'list_u' column
                if 'list_u' in df and df['list_u'][0] > 0.01:
                    print(f"Document with 'list_u[0]' > 5: {csv_file}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

import os
import pandas as pd
import glob

# # The path format with placeholders for the variable part (1, 2, or 3)
#
# path_format = "/home/bob/project_game_theory_linux/SpiderProject_KatherLab/Results_nsclc_paper_3/PR_exp_K_sigma_0/{}/Sigma_0/Exponential/4_November"
# variables = ['1', '2', '3','4']  # The variable part of the path that changes
#
# # Loop over the possible variable values to find the correct path
# for variable in variables:
#     # Generate the path
#     path = path_format.format(variable)
#     print(f"Looking in: {path}")  # Debug: print the path being searched
#
#     if not os.path.exists(path):
#         print(f"Path does not exist: {path}")
#         continue
#
#     # List all CSV files in this directory
#     csv_files = glob.glob(os.path.join(path, "*.csv"))
#     print(f"csv:{csv_files}")
#     for csv_file in csv_files:
#         try:
#             # Read the CSV file
#             df = pd.read_csv(csv_file)
#             # Debug: print the name of the file being processed
#             print(f"Processing file: {csv_file}")
#             # Check the first value in the 'list_u' column
#             if 'list_u[0](u_values)' in df.columns and df['list_u[0](u_values)'].iloc[0] > 5:
#                 print(f"Document with 'list_u[0]' > 5: {csv_file}")
#             else:
#                 # If the condition is not met, print that it's not met for debugging
#                 print(f"No matching 'list_u[0]' > 5 in: {csv_file}")
#         except Exception as e:
#             print(f"Error reading {csv_file}: {e}")

import os
import pandas as pd
import glob
#
# # The path format with placeholders for the variable part (1, 2, 3, 4)
# path_format = "/home/bob/project_game_theory_linux/SpiderProject_KatherLab/Results_nsclc_paper_3/PR_exp_K_sigma_0/{}/Sigma_0/Exponential/4_November"
# variables = ['1', '2', '3', '4']  # The variable part of the path that changes
#
# # Loop over the possible variable values to find the correct path
# for variable in variables:
#     # Generate the base path for the current variable
#     base_path = path_format.format(variable)
#     print(f"Looking in: {base_path}")  # Debug: print the base path being searched
#
#     if not os.path.exists(base_path):
#         print(f"Path does not exist: {base_path}")
#         continue
#
#     # Find all subdirectories in the base path
#     subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     for subdir in subdirs:
#         subdir_path = os.path.join(base_path, subdir)
#         # List all CSV files in the subdirectory
#         csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
#         print(f"CSV files in {subdir_path}: {csv_files}")  # Debug: print the CSV files found
#
#         for csv_file in csv_files:
#             try:
#                 # Read the CSV file
#                 df = pd.read_csv(csv_file)
#                 # Debug: print the name of the file being processed
#                 print(f"Processing file: {csv_file}")
#
#                 # Assuming 'list_u[0](u_values)' is the column name, replace it with the actual column name
#                 column_name = 'list_u[0](u_values)'
#                 if column_name in df.columns:
#                     # Check the first value in the specified column
#                     if df[column_name].iloc[0] > 5:
#                         print(f"Document with '{column_name}' > 5: {csv_file}")
#                     else:
#                         print(f"No matching '{column_name}' > 5 in: {csv_file}")
#                 else:
#                     print(f"Column '{column_name}' not found in: {csv_file}")
#             except Exception as e:
#                 print(f"Error reading {csv_file}: {e}")
#
# import os
# import pandas as pd
# import glob
#
# # The path format with placeholders for the variable part (1, 2, 3, 4)
# path_format = "/home/bob/project_game_theory_linux/SpiderProject_KatherLab/Results_nsclc_paper_3/PR_exp_K_sigma_0/{}/Sigma_0/Exponential/4_November"
# variables = ['1', '2', '3', '4']  # The variable part of the path that changes
#
# # Loop over the possible variable values to find the correct path
# for variable in variables:
#     # Generate the path
#     path = path_format.format(variable)
#     print(f"Looking in: {path}")  # Debug: print the path being searched
#
#     if not os.path.exists(path):
#         print(f"Path does not exist: {path}")
#         continue
#
#         # Find all subdirectories in the base path
#     subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
#     for subdir in subdirs:
#         subdir_path = os.path.join(base_path, subdir)
#         # List all CSV files in the subdirectory
#         csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
#         print(f"CSV files in {subdir_path}: {csv_files}")  # Debug: print the CSV files found
#
#         for csv_file in csv_files:
#             try:
#                 # Read the CSV file
#                 df = pd.read_csv(csv_file)
#                 # Debug: print the headers
#                 print(f"Headers in {csv_file}: {df.columns.tolist()}")
#
#                 # Define the column index for 'list_u[0](u_values)', which appears to be the 8th column, so index 7 (0-based index)
#                 column_index = 7
#                 # Check the first value in the 8th column
#                 if df.iloc[0, column_index] > 5:
#                     print(f"Document with 'list_u[0](u_values)' > 5: {csv_file}")
#                 else:
#                     print(f"No matching 'list_u[0](u_values)' > 5 in: {csv_file}")
#             except Exception as e:
#                 print(f"Error reading {csv_file}: {e}")
#
#

#####################
import os
import pandas as pd
import glob

# The path format with placeholders for the variable part (1, 2, 3, 4)
base_path_format = "/home/bob/project_game_theory_linux/SpiderProject_KatherLab/Results_nsclc_paper_3/PR_exp_K_sigma_0/{}/Sigma_0/Exponential/4_November"
variables = ['1', '2', '3', '4']  # The variable part of the path that changes

# Loop over the possible variable values to find the correct path
for variable in variables:
    # Generate the base path for the current variable
    base_path = base_path_format.format(variable)
    print(f"Looking in: {base_path}")  # Debug: print the base path being searched

    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        continue

    # List all subdirectories within the base path
    subdirectories = [os.path.join(base_path, d) for d in os.listdir(base_path) if
                      os.path.isdir(os.path.join(base_path, d))]

    # Loop through each subdirectory
    for subdir in subdirectories:
        print(f"Looking in subdirectory: {subdir}")  # Debug: print the subdirectory being searched
        # List all CSV files in this subdirectory
        csv_files = glob.glob(os.path.join(subdir, "*.csv"))
        for csv_file in csv_files:
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)

                # Convert the 'list_u[0](u_values)' column to numeric, errors='coerce' will replace non-numeric values with NaN
                df['list_u[0](u_values)'] = pd.to_numeric(df['list_u[0](u_values)'], errors='coerce')

                # Check the first value in the 'list_u[0](u_values)' column, after conversion to numeric
                if df['list_u[0](u_values)'].iloc[0] > 1:
                    print(f"Document with 'list_u[0](u_values)' > 5: {csv_file}")
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")

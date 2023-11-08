# import os
# import pandas as pd
# import glob
#
# # Base directory containing the folders and files
# base_dir = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\project_game_theory_linux\SpiderProject_KatherLab\Parallel_code_fixed_grouping_total_run"
#
# # Lists to hold the variants of paths and file types
# treatments = ['Chemo', 'Immuno']
# conditions = ['Inc', 'Dec']
# sizes = ['Size1', 'Size2', 'Size3', 'Size4']
#
# # Initialize an empty DataFrame to hold all combined data
# combined_df = pd.DataFrame()
#
# # Iterate through the different combinations of paths and filenames
# for treatment in treatments:
#     for condition in conditions:
#         # Construct the directory path
#         dir_path = os.path.join(base_dir, treatment, condition, 'Total')
#
#         # Check if the directory exists before proceeding
#         if not os.path.exists(dir_path):
#             print(f"Directory does not exist: {dir_path}")
#             continue
#
#         for size in sizes:
#             # Construct the file pattern
#             file_pattern = f"Total_study_{treatment}_{size}_{condition}_results.txt"
#             search_pattern = os.path.join(dir_path, file_pattern)
#
#             # Use glob to find all files matching the pattern
#             for file_path in glob.glob(search_pattern):
#                 # Read the .txt file into a DataFrame
#                 temp_df = pd.read_csv(file_path, delimiter='\t', header=0, dtype=str)
#
#                 # Extract additional information from file name
#                 temp_df['Treatment'] = treatment
#                 temp_df['Condition'] = condition
#                 temp_df['Size'] = size
#
#                 # Combine with the main DataFrame
#                 combined_df = pd.concat([combined_df, temp_df], ignore_index=True)
# #Options
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# # Display the combined DataFrame
# print(combined_df)
#
# # You may want to save the combined DataFrame to a new .csv file
# #combined_df.to_csv(r"C:\Users\Shade\Desktop\combined_data.csv", index=False)
#
# # You may want to save the combined DataFrame to a new Excel file
# excel_file_path = r"C:\Users\Shade\Desktop\combined_data.xlsx"
# combined_df.to_excel(excel_file_path, index=False)


###Comma problem:
import os
import pandas as pd
import glob
import csv

# Base directory containing the folders and files
base_dir = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\project_game_theory_linux\SpiderProject_KatherLab\Parallel_code_fixed_grouping_total_run"

# Lists to hold the variants of paths and file types
treatments = ['Chemo', 'Immuno']
conditions = ['Inc', 'Dec']
sizes = ['Size1', 'Size2', 'Size3', 'Size4']

# Initialize an empty DataFrame to hold all combined data
combined_df = pd.DataFrame()

# Iterate through the different combinations of paths and filenames
for treatment in treatments:
    for condition in conditions:
        # Construct the directory path
        dir_path = os.path.join(base_dir, treatment, condition, 'Total')

        # Check if the directory exists before proceeding
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
            continue

        for size in sizes:
            # Construct the file pattern
            file_pattern = f"Total_study_{treatment}_{size}_{condition}_results.txt"
            search_pattern = os.path.join(dir_path, file_pattern)

            # Use glob to find all files matching the pattern
            for file_path in glob.glob(search_pattern):
                # Read the .txt file into a DataFrame
                temp_df = pd.read_csv(file_path, delimiter=',', header=0, dtype=str, quoting=csv.QUOTE_MINIMAL)

                # Extract additional information from file name
                temp_df['Treatment'] = treatment
                temp_df['Condition'] = condition
                temp_df['Size'] = size

                # Combine with the main DataFrame
                combined_df = pd.concat([combined_df, temp_df], ignore_index=True)


#Options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Display the combined DataFrame
print(combined_df)

# Save the combined DataFrame to a new Excel file
excel_file_path = r"C:\Users\Shade\Desktop\combined_data.xlsx"
combined_df.to_excel(excel_file_path, index=False)

print(f"Combined data saved to Excel file at: {excel_file_path}")

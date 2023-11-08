import os
import pandas as pd
import numpy as np
import openpyxl
def analyze_files(base_path):
    # Define the different variations of the file paths
    treatments = ['Chemo', 'Immuno']
    variations = ['Inc', 'Dec']
    sizes = ['Size1', 'Size2', 'Size3', 'Size4']

    # Create a list to store the results of the analysis
    results = []

    # Iterate through all possible combinations of file paths
    for treatment in treatments:
        for variation in variations:
            for size in sizes:
                # Construct the file name based on the current combination
                file_name = f"study_{treatment}_{size}_{variation}_results.txt"
                file_path = os.path.join(base_path, treatment, variation, 'Index_removed', file_name)

                # Check if the file exists
                if os.path.exists(file_path):
                    # Read the data from the file into a pandas DataFrame
                    df = pd.read_csv(file_path)
                    # Analyze the file and store the results in the results list
                    results.append(analyze_file(df, file_path, treatment, size, variation))
                else:
                    print(f"File not found: {file_path}")

    # Convert the results list into a pandas DataFrame for better visualization
    results_df = pd.DataFrame(results)
    # Print the results DataFrame
    print(results_df)

def analyze_file(df, file_name, treatment, size, variation):
    # Create a dictionary to store the results of the analysis for this file
    result = {
        "File name": file_name,
        "Treatment": treatment,
        "Size": size,
        "Variation": variation,
        "Number of Rows": len(df)
    }

    # Calculate the range and percentage change for each numerical column in the DataFrame
    for column in df.columns[3:]:  # Skipping the first three columns as they are not numerical
        min_val = df[column].min()
        max_val = df[column].max()
        num_unique_values = len(df[column].unique())
        percentage_change = ((num_unique_values - 1) / (len(df) - 1)) * 100 if len(df) > 1 else 0

        result[f"{column} Range"] = f"{min_val} to {max_val}"
        result[f"{column} % Change"] = f"{percentage_change:.2f}%"

    return result

if __name__ == "__main__":
    # Specify the base path to the directory containing the text files
    base_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\project_game_theory_linux\SpiderProject_KatherLab\Parallel_code_3_Fixed_groups_1to4"
    #C, Setting options
    pd.set_option('display.max_columns', None)  # Ensure all columns are displayed
    pd.set_option('display.expand_frame_repr', False)  # Prevent the printout from line-wrapping

    # Run the file analysis
    results_df = analyze_files(base_path)

    #results_df = pd.DataFrame(results_df)
    #Printing it (just to be sure)
    #print(results_df)
    # Specify the Excel file name
    #excel_file_name = "results_table_bootstrapping.xlsx"

    # Write the DataFrame to an Excel file
    #results_df.to_excel(excel_file_name, index=False, engine='openpyxl')

    #print(f"Results table saved to {excel_file_name}")

import os
import pandas as pd
import openpyxl


def analyze_file(df, file_name, treatment, size, variation):
    result = {
        "File Name": os.path.basename(file_name),  # Use just the file name, not the full path
        "Treatment": treatment,
        "Size": size,
        "Variation": variation,
        "Number of Rows": len(df)
    }

    for column in df.columns[3:]:  # Assuming you want to skip the first three columns
        min_val = df[column].min()
        max_val = df[column].max()
        num_unique_values = len(df[column].unique())
        percentage_change = ((num_unique_values - 1) / (len(df) - 1)) * 100 if len(df) > 1 else 0

        result[f"{column} Range"] = f"{min_val} to {max_val}"
        result[f"{column} % Change"] = f"{percentage_change:.2f}%"

    return result


def analyze_files(base_path):
    treatments = ['Chemo', 'Immuno']
    variations = ['Inc', 'Dec']
    sizes = ['Size1', 'Size2', 'Size3', 'Size4']

    results = []

    for treatment in treatments:
        for variation in variations:
            for size in sizes:
                file_name = f"study_{treatment}_{size}_{variation}_results.txt"  # Adjusted file name pattern
                file_path = os.path.join(base_path, treatment, variation, 'Index_removed', file_name)

                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)
                    results.append(analyze_file(df, file_path, treatment, size, variation))
                else:
                    print(f"File not found: {file_path}")

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    base_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\project_game_theory_linux\SpiderProject_KatherLab\Parallel_code_3_Fixed_groups_1to4"
    results_df = analyze_files(base_path)

    if not results_df.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.expand_frame_repr', False)

        print(results_df)

        excel_file_name = "results_table_bootstrapping_fixed_grouping.xlsx"
        results_df.to_excel(excel_file_name, index=False, engine='openpyxl')
        print(f"Results table saved to {excel_file_name}")
    else:
        print("No data found.")

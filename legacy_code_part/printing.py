# Define the target directory and file path
import os
import sys
import pickle

dataset_path = r"C:\Users\Shade\Desktop\Master\Project Game Theory Code\Downloaded file\Edited\SpiderProject_KatherLab"  # Use a raw string for the path
sys.path.insert(0,dataset_path)


target_dir = os.path.join(dataset_path, "Param_size1_control")

file_path = os.path.join(target_dir, "study_size1_param_control" + '.pkl')

# Load the contents of the pickle file and print it
with open(file_path, 'rb') as f:
    loaded_result = pickle.load(f)

print(loaded_result)
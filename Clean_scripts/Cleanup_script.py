#Improtting Libraries
import shutil
from pathlib import Path
import time

#Letting Parralel script getting ahead
time.sleep(360)
def clean_temp_gk_model_folders(directory="/tmp", keep_count=15):
    try:
        print("Running cleanup script...")  # Confirm that the script is running
        temp_directory = Path(directory)
        temp_gk_folders = [f for f in temp_directory.iterdir() if f.is_dir() and f.name.startswith("tmp") and "gk_model" in f.name]
        
        print("Gekko Model folders:", temp_gk_folders)  # List all Gekko model folders

        if len(temp_gk_folders) > keep_count:
            sorted_folders = sorted(temp_gk_folders, key=lambda x: x.stat().st_mtime)
            folders_to_delete = sorted_folders[:-keep_count]
            
            print("Folders to delete (sorted by oldest):", folders_to_delete)  # List all folders to be deleted
            
            for folder in folders_to_delete:
                shutil.rmtree(folder)
                print(f"Deleted: {folder}")

        else:
            print("Not enough folders to trigger deletion. No folders deleted.")

    except Exception as e:
        print("An error occurred:", str(e))

#Manualy change how long you want to run
for _ in range(222240):  # 120 intervals of 10 minutes in 20 hours
    clean_temp_gk_model_folders()
    time.sleep(120)  # Wait for 10 minutes


from pathlib import Path

from fitting.load_data import get_current_data

data_df = get_current_data()
for _, row in data_df.iterrows():
    all_files_exist = True

    file_path_to_check = Path(row["image_path"])
    if file_path_to_check.exists():
        continue
    else:
        print(f"{file_path_to_check} does not exist.")
        all_files_exist = False
        break
if all_files_exist == True:
    print("All files exist")
else:
    print("Found Missing files")

import os
import shutil

source_folder = 'integrals'
base_output_folder = 'dataset_pix2pix/input'

if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder)

folders = {"0.1": "B", "0.8": "C", "1.6": "D"}
for focal_length, folder_name in folders.items():
    folder_path = os.path.join(base_output_folder, folder_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def sort_files_by_focal_length():
    for filename in os.listdir(source_folder):
        if filename.startswith("0_") and "_integral_f-" in filename:
            parts = filename.split('_')
            number = parts[1]
            focal_length = parts[3].replace("f-", "").replace(".png", "")

            if focal_length in folders:
                destination_folder = os.path.join(base_output_folder, folders[focal_length])
                new_filename = f"{number}.png"
                shutil.move(os.path.join(source_folder, filename), os.path.join(destination_folder, new_filename))

sort_files_by_focal_length()
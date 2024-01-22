import os
import shutil

source_folder = 'groundtruths'
output_folder = 'dataset_pix2pix/groundtruth'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def rename_and_copy_files():
    for filename in os.listdir(source_folder):
        if filename.startswith("0_") and filename.endswith("_GT_pose_0_thermal.png"):
            number = filename.split('_')[1]
            new_filename = f"{number}.png"
            shutil.copy(os.path.join(source_folder, filename), os.path.join(output_folder, new_filename))

rename_and_copy_files()

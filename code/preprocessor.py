import os
import shutil
import subprocess

def prepare_groundtruths(source_folder='groundtruths', output_folder='dataset_pix2pix/groundtruth'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(source_folder):
        if filename.startswith("0_") and filename.endswith("_GT_pose_0_thermal.png"):
            number = filename.split('_')[1]
            new_filename = f"{number}.png"
            shutil.copy(os.path.join(source_folder, filename), os.path.join(output_folder, new_filename))

def prepare_integrals(source_folder='integrals', base_output_folder='dataset_pix2pix/input'):
    if not os.path.exists(base_output_folder):
        os.makedirs(base_output_folder)

    folders = {"0.1": "B", "0.8": "C", "1.6": "D"}
    for focal_length, folder_name in folders.items():
        folder_path = os.path.join(base_output_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    for filename in os.listdir(source_folder):
        if filename.startswith("0_") and "_integral_f-" in filename:
            parts = filename.split('_')
            number = parts[1]
            focal_length = parts[3].replace("f-", "").replace(".png", "")

            if focal_length in folders:
                destination_folder = os.path.join(base_output_folder, folders[focal_length])
                new_filename = f"{number}.png"
                shutil.copy(os.path.join(source_folder, filename), os.path.join(destination_folder, new_filename))

def split_files(src_folder='dataset_pix2pix', train_size=8800):
    subfolders = ['groundtruth', 'input/B', 'input/C', 'input/D']
    for subfolder in subfolders:
        full_src_folder = os.path.join(src_folder, subfolder)
        train_folder = full_src_folder + '/train'
        val_folder = full_src_folder + '/val'

        if not os.path.exists(full_src_folder):
            print(f"Source folder {full_src_folder} not found. Skipping split for {subfolder}.")
            continue

        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(val_folder, exist_ok=True)

        files = [f for f in os.listdir(full_src_folder) if os.path.isfile(os.path.join(full_src_folder, f))]

        for file in files[:train_size]:
            shutil.copy(os.path.join(full_src_folder, file), train_folder)

        for file in files[train_size:]:
            shutil.copy(os.path.join(full_src_folder, file), val_folder)

        print(f"Files in {subfolder} copied successfully.")

def delete_files_in_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            print(f"Skipped directory: {item_path}")

def run_combine():
    command = [
        'python', 'pytorch-CycleGAN-and-pix2pix/datasets/combine_A_and_B_and_C_and_D.py',
        '--fold_A', 'dataset_pix2pix/groundtruth',
        '--fold_B', 'dataset_pix2pix/input/B',
        '--fold_C', 'dataset_pix2pix/input/C',
        '--fold_D', 'dataset_pix2pix/input/D',
        '--fold_ABCD', 'dataset_pix2pix/combined'
    ]

    subprocess.run(command)

prepare_groundtruths()
prepare_integrals()
split_files()
directories = [
    "dataset_pix2pix/groundtruth",
    "dataset_pix2pix/input/B",
    "dataset_pix2pix/input/C",
    "dataset_pix2pix/input/D"
]
for directory in directories:
    delete_files_in_directory(directory)
run_combine()
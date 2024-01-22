import os
import shutil


def copy_files(src_folder, train_folder, val_folder, train_size):
    if not os.path.exists(src_folder):
        print(f"Source folder {src_folder} not found.")
        return
    
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)

    files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]

    for file in files[:train_size]:
        shutil.copy(os.path.join(src_folder, file), train_folder)

    for file in files[train_size:]:
        shutil.copy(os.path.join(src_folder, file), val_folder)

    print("Files copied successfully.")

src_folder = 'dataset_pix2pix/groundtruth'
train_folder = src_folder + '/train'
val_folder = src_folder + '/val'
train_size = 8800

copy_files(src_folder, train_folder, val_folder, train_size)
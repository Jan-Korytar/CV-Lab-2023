import os
from PIL import Image

source_folder = 'integrals'
base_output_folder = 'train_test'

if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder)

image_files = {}

for filename in os.listdir(source_folder):
    if filename.startswith("0_") and "_integral_f-" in filename:
        parts = filename.split('_')
        number = parts[1]
        focal_length = parts[3].replace("f-", "").replace(".png", "")
        if focal_length in ["0.1", "0.8", "1.6"]:
            if number not in image_files:
                image_files[number] = {'B': None, 'C': None, 'D': None}
            if focal_length == "0.1":
                image_files[number]['B'] = os.path.join(source_folder, filename)
            elif focal_length == "0.8":
                image_files[number]['C'] = os.path.join(source_folder, filename)
            elif focal_length == "1.6":
                image_files[number]['D'] = os.path.join(source_folder, filename)

def concatenate_and_save_images():
    for number, images in image_files.items():
        imgs_to_concat = [Image.open(images[fl]) for fl in ['B', 'C', 'D'] if images[fl] is not None]
        if len(imgs_to_concat) == 3:
            concatenated_img = Image.new('RGB', (sum(img.width for img in imgs_to_concat), imgs_to_concat[0].height))
            x_offset = 0
            for img in imgs_to_concat:
                concatenated_img.paste(img, (x_offset, 0))
                x_offset += img.width
            concatenated_img.save(os.path.join(base_output_folder, f"{number}.png"))

concatenate_and_save_images()
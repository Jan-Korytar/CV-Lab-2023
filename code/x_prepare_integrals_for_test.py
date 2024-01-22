import os
from PIL import Image

# Source and destination folder paths
source_folder = 'integrals'
base_output_folder = 'dataset_pix2pix_test/input/test'

# Create the base output folder if it doesn't exist
if not os.path.exists(base_output_folder):
    os.makedirs(base_output_folder)

# Dictionary to track images by number
image_files = {}

# Iterate through files and group them by number
for filename in os.listdir(source_folder):
    if filename.startswith("1_") and "_integral_f-" in filename:
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

# Function to concatenate and save images
def concatenate_and_save_images():
    for number, images in image_files.items():
        # Load images if they exist and concatenate them horizontally
        imgs_to_concat = [Image.open(images[fl]) for fl in ['B', 'C', 'D'] if images[fl] is not None]
        if len(imgs_to_concat) == 3:
            concatenated_img = Image.new('RGB', (sum(img.width for img in imgs_to_concat), imgs_to_concat[0].height))
            x_offset = 0
            for img in imgs_to_concat:
                concatenated_img.paste(img, (x_offset, 0))
                x_offset += img.width
            # Save the concatenated image
            concatenated_img.save(os.path.join(base_output_folder, f"{number}.png"))

# Execute the function
concatenate_and_save_images()

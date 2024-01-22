import subprocess
import os
import argparse
from PIL import Image

def concatenate_and_save_images(image_path_01, image_path_08, image_path_16, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_01 = Image.open(image_path_01)
    img_08 = Image.open(image_path_08)
    img_16 = Image.open(image_path_16)

    concatenated_img = Image.new('RGB', (img_01.width + img_08.width + img_16.width, img_01.height))
    concatenated_img.paste(img_01, (0, 0))
    concatenated_img.paste(img_08, (img_01.width, 0))
    concatenated_img.paste(img_16, (img_01.width + img_08.width, 0))

    output_path = os.path.join(output_folder, "combined_image.png")
    concatenated_img.save(output_path)

def run_test_script():
    command = [
        'python', 'pytorch-CycleGAN-and-pix2pix/test.py',
        '--dataroot', 'temp_test',
        '--name', 'weights',
        '--checkpoints_dir', './',
        '--results_dir', 'test_results',
        '--model', 'test',
        '--direction', 'BtoA',
        '--dataset_mode', 'single',
        '--norm', 'batch',
        '--netG', 'unet_256',
        '--input_nc', '9',
        '--eval',
        '--epoch', '250',
        '--num_test', '1',
        '--batch_size', '8'
    ]

    subprocess.run(command)

def upscale_image(image_path, size=(512, 512)):
    with Image.open(image_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        resized_img.save(image_path)

def main():
    import sys
    if len(sys.argv) != 4:
        print("Usage: python test.py <image_path_length_0.1> <image_path_length_0.8> <image_path_length_1.6>")
        sys.exit(1)

    image_path_01 = sys.argv[1]
    image_path_08 = sys.argv[2]
    image_path_16 = sys.argv[3]

    concatenate_and_save_images(image_path_01, image_path_08, image_path_16, 'temp_test')
    run_test_script()
    image_path = 'test_results/weights/test_250/images/combined_image_fake.png'
    upscale_image(image_path)

if __name__ == "__main__":
    main()
import subprocess
import os
import argparse
from PIL import Image
import glob
from skimage.metrics import structural_similarity as ssim
import numpy as np

def run_eval_script():
    command = [
        'python', 'pytorch-CycleGAN-and-pix2pix/test.py',
        '--dataroot', 'train_test',
        '--name', 'weights',
        '--checkpoints_dir', './',
        '--results_dir', 'result',
        '--model', 'test',
        '--direction', 'BtoA',
        '--dataset_mode', 'single',
        '--norm', 'batch',
        '--netG', 'unet_256',
        '--input_nc', '9',
        '--eval',
        '--epoch', '250',
        '--num_test', '11000',
        '--batch_size', '8'
    ]
    subprocess.run(command)

def resize_image(image_path, size=(256, 256), convert_to_rgb=False):
    with Image.open(image_path) as img:
        if convert_to_rgb:
            img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img

def calculate_ssim(image1, image2):
    return ssim(image1, image2, multichannel=True, channel_axis=-1)

def calculate_average_ssim():
    # find all model output images
    model_images = glob.glob('result/weights/test_250/images/*_fake.png')

    # find all ground truth images
    ground_truth_images = glob.glob('groundtruths/*0_*_GT_pose_0_thermal.png')

    # extract numbers and map model outputs to ground truths
    model_ground_truth_pairs = {}
    for model_image in model_images:
        number = os.path.basename(model_image).split('_')[0]
        corresponding_gt = [gt for gt in ground_truth_images if f'0_{number}_' in gt]
        if corresponding_gt:
            model_ground_truth_pairs[model_image] = corresponding_gt[0]

    # calculate SSIM for each pair
    ssim_scores = {}
    total_ssim_score = 0
    matched_images_count = 0
    for model_image, gt_image in model_ground_truth_pairs.items():
        model_img = resize_image(model_image)
        gt_img = resize_image(gt_image, convert_to_rgb=True)

        model_img_np = np.array(model_img)
        gt_img_np = np.array(gt_img)

        score = calculate_ssim(model_img_np, gt_img_np)
        ssim_scores[model_image] = score
        total_ssim_score += score
        matched_images_count += 1

    average_ssim_score = total_ssim_score / matched_images_count
    return average_ssim_score

def main():
    run_eval_script()
    avg_ssim = calculate_average_ssim()
    print(f"\nAverage SSIM score: {avg_ssim}")

if __name__ == "__main__":
    main()
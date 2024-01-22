import glob
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import os
import numpy as np

def resize_image(image_path, size=(256, 256), convert_to_rgb=False):
    with Image.open(image_path) as img:
        if convert_to_rgb:
            img = img.convert('RGB')
        img = img.resize(size, Image.Resampling.LANCZOS)
        return img

def calculate_ssim(image1, image2):
    return ssim(image1, image2, multichannel=True, channel_axis=-1)

# find all model output images
model_images = glob.glob('results/thermal_combined_pix2pix/test_250/images/*_fake.png')

# find all ground truth images
ground_truth_images = glob.glob('groundtruths/*1_*_GT_pose_0_thermal.png')

# extract numbers and map model outputs to ground truths
model_ground_truth_pairs = {}
for model_image in model_images:
    number = os.path.basename(model_image).split('_')[0]
    corresponding_gt = [gt for gt in ground_truth_images if f'1_{number}_' in gt]
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
print(f"\nAverage SSIM score: {average_ssim_score}")
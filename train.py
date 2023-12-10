import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from cvlab_dataset import cvlab_dataset
from model import UNet
import glob
from tqdm import tqdm
from natsort import os_sorted
import os
import shutil

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your dataset class and DataLoader here (replace with your actual implementation)
directory_to_img_gt = r'C:\my files\cv_lab\Part1'
directory_to_integral_img = r'C:\my files\cv_lab\integral_images'

#beware os sorted, it can fuck up your sorted order, works in windows tho
train_images = os_sorted(glob.glob(f"{directory_to_integral_img}/*.png"))
gt_images = os_sorted(glob.glob(f"{directory_to_img_gt}/*_GT_pose_0_thermal.png"))
gt_images = gt_images[:int(len(train_images)//3)]

'''
for image in train_images.copy():
    img_base = image.split('-')[0]
    'C:\\my files\\cv_lab\\integral_images\\0_0_integral_f-0.21.png'
    if os.path.join(directory_to_img_gt, os.path.basename(img_base.replace('_integral_f', '_GT_pose_0_thermal.png'))) not in gt_images:
        train_images.remove(image)

for gt in gt_images.copy():
    continue # hope and pray all of them have their own ground truth :)
    #if gt.replace('_GT_pose_0_thermal.png', '_pose_5_thermal.png') not in train_images:
    #    gt_images.remove(gt)
'''
dataset = cvlab_dataset(train_images[:], gt_images[:])
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = UNet(3, 64, 3, 1).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs and early stopping parameters
num_epochs = 50
patience = 10
best_loss = float('inf')
counter = 0
losses = []


if os.path.exists('imgs'):
        # Remove the existing directory
        shutil.rmtree('imgs')

    # Create a new directory with the same name
os.makedirs('imgs')

# Training loop
for epoch in tqdm(range(num_epochs), desc='Epoch:'):
    model.train()
    running_loss = 0.0

    for inputs, labels in (tq_bar := tqdm(dataloader, desc='running_loss')):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        tq_bar.set_description(f'Running_loss: {loss.item()}', refresh=True)

    average_loss = running_loss / len(dataloader)
    losses.append(average_loss)
    #TODO Validation
    # Perform validation and compute validation_loss here

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

    with torch.no_grad():
        if (epoch + 1) % 5 == 0 or epoch <= 10:
            model.eval()
            sample_output = model(inputs[:1])  # Assuming batch size is at least 1

            # Assuming sample_output is a single-channel image tensor, adjust as needed
            sample_output_image = F.to_pil_image(sample_output[0].cpu())
            sample_output_image.save(f'imgs/epoch_{epoch + 1}_output.png')
            sample_output_image = F.to_pil_image(labels[0].cpu())
            sample_output_image.save(f'imgs/epoch_{epoch + 1}_gt.png')
            sample_output_image = F.to_pil_image(inputs[0, 0].cpu())
            sample_output_image.save(f'imgs/epoch_{epoch + 1}_input.png')


    '''# Early stopping check
    if validation_loss < best_loss:
        best_loss = validation_loss
        counter = 0
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping. Training halted.")
        break'''

print(losses)
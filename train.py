import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from cvlab_dataset import cvlab_dataset
from model import UNet
import glob
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define your dataset class and DataLoader here (replace with your actual implementation)
directory_to_img = 'C:\my files\cv_lab\Part1'
train_images = sorted(glob.glob(f"{directory_to_img}/*_pose_5_thermal.png"))
gt_images = sorted(glob.glob(f"{directory_to_img}/*_GT_pose_0_thermal.png"))
for image in train_images.copy():
    if image.replace('_pose_5_thermal.png', '_GT_pose_0_thermal.png') not in gt_images:
        train_images.remove(image)
for gt in gt_images.copy():
    if gt.replace('_GT_pose_0_thermal.png', '_pose_5_thermal.png') not in train_images:
        gt_images.remove(gt)

dataset = cvlab_dataset(train_images[:800], gt_images[:800], None)
dataloader = DataLoader(dataset, batch_size=8)

model = UNet(3, 16, 1, 1).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs and early stopping parameters
num_epochs = 10
patience = 10
best_loss = float('inf')
counter = 0
losses = []

# Training loop
for epoch in (range(num_epochs)):
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()


    average_loss = running_loss / len(dataloader)

    #TODO Validation
    # Perform validation and compute validation_loss here

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {average_loss:.4f}")

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
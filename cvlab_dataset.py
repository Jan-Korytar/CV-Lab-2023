from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image


class cvlab_dataset(Dataset):

    def __init__(self, train_images, gt_images, size=128):
        self.train_images = train_images
        self.gt_images = gt_images
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,),)
        ])
        self.transform_gt = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()])
        if len(self.train_images) != len(self.gt_images):
            raise ValueError("The length of train_images and gt_images is not the same")

    def __len__(self):
        return len(self.train_images)

    def __getitem__(self, index):
        image = self.transform(Image.open(self.train_images[index]))
        gt_image = self.transform_gt(Image.open(self.gt_images[index]))
        return image, gt_image

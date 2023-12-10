from torchvision import models
from torchsummary import summary
from model import UNet

model = UNet(3, 64, 3, 1)
summary(model, (3, 512, 512))

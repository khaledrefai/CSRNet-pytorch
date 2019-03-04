#importing libraries
from model import CSRNet
import torch
from image import *
import PIL.Image as Image

from torchvision import datasets, transforms
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
                  ])

                      
model = CSRNet()


#loading the trained weights
checkpoint = torch.load('0model_best.pth.tar',map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

img = transform(Image.open('test3.jpg').convert('RGB'))#.cuda()
output = model(img.unsqueeze(0))
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm

from ipywidgets import IntProgress

device = "cuda:4" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

data_dir = './IDD_Segmentation/leftImg8bit/val/'

scenes = os.listdir(data_dir)
dict_scenes = {}

for scene in scenes:
    for img in os.listdir(data_dir + scene):
        dict_scenes[img] = scene


num_classes = 27

class UNet(nn.Module):
    
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)
        
        
    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block
    
    def forward(self, X):
        contracting_11_out = self.contracting_11(X) # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(contracting_11_out) # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(contracting_12_out) # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(contracting_21_out) # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(contracting_22_out) # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(contracting_31_out) # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(contracting_32_out) # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(contracting_41_out) # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out) # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out) # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat((expansive_11_out, contracting_41_out), dim=1)) # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(expansive_12_out) # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat((expansive_21_out, contracting_31_out), dim=1)) # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(expansive_22_out) # [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(torch.cat((expansive_31_out, contracting_21_out), dim=1)) # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_41_out = self.expansive_41(expansive_32_out) # [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(torch.cat((expansive_41_out, contracting_11_out), dim=1)) # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out) # [-1, num_classes, 256, 256]
        # output_out = torch.softmax(output_out, dim=1)
        return output_out

model_path = './IDD_Segmentation/checkpoints/UNet_19.pkl' 
model = UNet(num_classes = num_classes).to(device)
model.load_state_dict(torch.load(model_path))

transform_ops = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.485, 0.56, 0.406),
                                                            std=(0.229, 0.224, 0.225))])

val_dir = './IDD_Segmentation/val/'
labels_dir = './IDD_Segmentation/gtAll/'

dump_dir = './IDD_Segmentation/val_predictions/'

val_img = os.listdir(val_dir)

iou_scores = []

for img in val_img:
    image = Image.open(val_dir+img).convert("RGB")
    img_shape = np.asarray(image).shape
    image = image.resize((256, 256))
    image = transform_ops(image)
    
    image = image.reshape(1,3,256,256)

    pred = model(image.to(device))
    pred = torch.argmax(pred, dim=1)

    pred = pred[0].cpu().detach().numpy().astype('uint8')
    #print("Before "+ str(pred.shape)) 
    pred = cv2.resize(pred,(img_shape[1], img_shape[0]) , cv2.INTER_NEAREST) 
    
    label = cv2.imread(labels_dir+img[:6]+'_gtFine_labellevel3Ids.png', cv2.IMREAD_GRAYSCALE)
    label = np.asarray(label)
    
    dump_scene = dict_scenes[img]

    if str(dump_scene) not in os.listdir(dump_dir):
        os.mkdir(dump_dir+dump_scene)

    cv2.imwrite(dump_dir + dump_scene + '/'+ img[:6] + '_gtFine_labellevel3Ids.png', pred)


    #print("After " + str(pred.shape))
    #print(label.shape)

    # intersection = np.logical_and(label, pred)
    # union = np.logical_or(label, pred)

    # iou_score = np.sum(intersection) / np.sum(union)
    # iou_scores.append(iou_score)
    #print(iou_score)    

# print("Validation average IOU score %0.4f"% (sum(iou_scores) / len(iou_scores)))








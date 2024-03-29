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

train_dir = './IDD_Segmentation/train/'
val_dir = './IDD_Segmentation/val/'
labels_dir = './IDD_Segmentation/gtAll/'

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)

print(len(train_fns), len(val_fns))

num_classes = 27

################################# DATA HANDLING ############################################

class ICVGIPDataset(Dataset):

    def __init__(self, image_dir, labels_dir):
        self.image_dir = image_dir
        self.labels_dir = labels_dir
        self.image_fns = os.listdir(image_dir)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        label_fp = os.path.join(self.labels_dir, image_fn[:6]+'_gtFine_labellevel3Ids.png')

        image = Image.open(image_fp).convert("RGB")
        image = image.resize((256, 256))
        image = np.asarray(image)
        
        labels = cv2.imread(label_fp, cv2.IMREAD_GRAYSCALE)

        labels = cv2.resize(labels, (256, 256), cv2.INTER_NEAREST)

        labels_temp = np.asarray(labels) 

        labels = torch.Tensor(np.asarray(labels)).long()

        image = self.transform(image)
        
        return image, labels

    def transform(self, image):
        transform_ops = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.485, 0.56, 0.406),
                                                                 std=(0.229, 0.224, 0.225))])
        
        return transform_ops(image)

################################# MODEL ############################################

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

################################# TRAINING LOOP ############################################

batch_size = 8 
val_batch_size = 8

epochs = 10 
lr = 0.01

dataset = ICVGIPDataset(train_dir, labels_dir)
data_loader = DataLoader(dataset, batch_size = batch_size)

val_dataset = ICVGIPDataset(val_dir, labels_dir)
val_data_loader = DataLoader(val_dataset, batch_size=val_batch_size)

model = UNet(num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)):
    epoch_loss = 0
    batch = 0
    for X,Y in tqdm(data_loader, total=len(data_loader), leave = False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
        del X, Y
        
        if(batch%50==0):
            print("Batch %d, Epoch %d"% (batch,epoch))

        batch+=1

    epoch_losses.append(epoch_loss/len(data_loader))
    model_name = "UNet_" + str(epoch) + ".pkl"
    torch.save(model.state_dict(), './IDD_Segmentation/checkpoints/' + model_name)



    inverse_transform = transforms.Compose([
            transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))  
        ])

    iou_scores = []

    ################################# VALIDATION ############################################

    for X,Y in tqdm(val_data_loader, total=len(val_data_loader), leave = False):
        X,Y = X.to(device), Y.to(device)
        Y_pred = model(X)
        Y_pred = torch.argmax(Y_pred, dim=1)
        
        for i in range(val_batch_size):
            try:
                landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
                label_class = Y[i].cpu().detach().numpy()
                label_class_predicted = Y_pred[i].cpu().detach().numpy()

                intersection = np.logical_and(label_class, label_class_predicted)
                union = np.logical_or(label_class, label_class_predicted)

                iou_score = np.sum(intersection) / np.sum(union)
                iou_scores.append(iou_score)
                #print(iou_score)
            except:
                pass 

    print("Validation average IOU score for epoch %d : %0.4f"% (epoch, (sum(iou_scores) / len(iou_scores))) )



fig, axes = plt.subplots(1,2, figsize=(10,5))
axes[0].plot(step_losses)
axes[1].plot(epoch_losses)

plt.savefig("./train_analysis.png")



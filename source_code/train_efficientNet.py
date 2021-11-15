
#Importing libraries
import random, shutil, os
import copy
from numpy.core.numeric import tensordot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from torch.utils.data.dataset import Dataset
#from extract_images import extract_images
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets, models

import sys


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Train EfficientNet on the images
#1. Create Datasets for training and validation
train_dir = 'tiles_1000_cnn_training_50clusters'
valid_dir = 'validation_tiles'
#Data transformation:
data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
#Create the data sets:        
train_dataset = datasets.ImageFolder(train_dir, data_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, data_transforms)

#2. Create Dataloaders for training (with shuffling) and validation
#Shuffle data before training: 
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers=0)
valid_loader = DataLoader(validation_dataset, batch_size = 8, shuffle = False, num_workers=0)

#3. Create the model with correct number of classes:
model = EfficientNet.from_name('efficientnet-b1')
#Unfreeze the model weights:
for params in model.parameters():
    params.requires_grad = True
#Add extra steps for last layer and correct number of classes (=1):
in_ftrs = model._fc.in_features
model._fc = nn.Linear(in_ftrs, 1)

#4. Train the model:
model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
#Since binary classification: This combines sigmoid and BCE and is more numerically stable:
# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html 
criterion = nn.BCEWithLogitsLoss()

#Training procedure from Steven's code:
def train(model, dataloaders, optimizer, criterion, n_epochs): 
        
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    val_acc_history = []

    for epoch_idx in range(n_epochs):

        for phase, dataloader in dataloaders.items():
            
            if phase == "TRAIN":
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_acc = 0.0

            with torch.set_grad_enabled(phase == "TRAIN"):

                for i, (inputs, y_true) in enumerate(dataloader):

                    inputs = inputs.to(DEVICE)
                    y_true = y_true.to(DEVICE)
                    #Unsqueeze to get correct dimensions:
                    y_true = y_true.unsqueeze(1)
                    
                    y_pred = model(inputs)
                    #Convert y_true to float
                    loss = criterion(y_pred, y_true.float())
                    
                    if phase == "TRAIN":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                    
                    y_true = y_true.detach().cpu().numpy()
                    y_pred = torch.round(torch.sigmoid(y_pred.detach().cpu()))
                    y_pred = y_pred.numpy()

                    running_loss += loss.item()
                    running_acc += metrics.accuracy_score(y_true, y_pred)
                    
            
            mean_loss = running_loss / len(dataloader)
            mean_acc = running_acc / len(dataloader)
            
            if phase == "VALID" and mean_acc > best_acc:
                best_acc = mean_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'VALID':
                val_acc_history.append(mean_acc)
            
            print("%s Epoch %i\t Loss: %.4f\t ACC: %.4f" % (phase, epoch_idx, mean_loss, mean_acc))

    print("Best val Acc: %.4f" % best_acc)
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

#Train the model for 25 epochs:
best_model, hist = train(model, {'TRAIN':train_loader, 'VALID': valid_loader}, optimizer, criterion, n_epochs = 25)

torch.save(best_model.state_dict(), 'tile_1000images_50clusters.pt')

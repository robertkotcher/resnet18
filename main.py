import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset import FocalLengthDataset

# source: https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch/notebook

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from resnet18 import ResNet_18

torch.manual_seed(17)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 1

######################################################################
# Gather data
######################################################################

# data_path = "./data"

# train = pd.read_csv(data_path + "/train.csv",dtype = np.float32)
# test = pd.read_csv(data_path + "/test.csv",dtype = np.float32)
# submission = pd.read_csv(data_path + "/sample_submission.csv")
# print("Train set shape:", train.shape)
# print("Test set shape:", test.shape)
## ~ 4000 training samples for each digit

######################################################################
# Prepare data - Pandas Dataframe makes this easy
######################################################################

# labels = train.label.values
# data = train.iloc[:, 1:].values / 255 # Normalization
# print("Labels shape: ", labels.shape)
# print("Dataset shape: ", data.shape)

# labels = torch.from_numpy(labels).type(torch.LongTensor)
# data = torch.from_numpy(data).view(data.shape[0], 1, 28, 28)

# train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size = 0.2, random_state = 42)


######################################################################
# CustomDataset - because we need to apply transformations
######################################################################
# class CustomTensorDataset(Dataset):

#     def __init__(self, data, labels=None, transform=None):
#         self.data = data
#         self.labels = labels
#         self.transform = transform

#     def __getitem__(self, index):
#         x = self.data[index]

#         if self.transform is not None:
#             x = self.transform(x)
#         if self.labels is not None:
#             y = self.labels[index]
#             return x, y
#         else:
#             return x

#     def __len__(self):
#         return self.data.size(0)

# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     # rotate and scale while keeping parallel relationships
#     transforms.RandomAffine(degrees=20, scale=(1.1, 1.1)),
#     # crop back to expected size
#     transforms.RandomCrop((28, 28), padding=2, pad_if_needed=True, fill=0, padding_mode='constant'),
#     transforms.ToTensor()
# ])


######################################################################
# DataLoader (and apply transforms to original set)
######################################################################
# trainset = ConcatDataset([
#     CustomTensorDataset(train_data, train_labels),
#     CustomTensorDataset(train_data, train_labels, transform=transform)
# ])
# valset = CustomTensorDataset(val_data, val_labels)

fd_train = FocalLengthDataset(data_dir="./data")
fd_val = FocalLengthDataset(data_dir="./data")

train_loader = DataLoader(fd_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(fd_val, batch_size=batch_size, shuffle=False)

model = ResNet_18(3, 1)

model.to(device)
next(model.parameters()).is_cuda

# ######################################################################
# # TrainingLoop
# ######################################################################

epochs = 5
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]: # Iterate over data
                
                inputs = transforms.functional.resize(inputs, (112, 112))
                inputs = inputs.to(device)

                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            
            if phase == 'val': # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)
                
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

model, _ = train_model(model, {"train": train_loader, "val": val_loader}, criterion, optimizer, epochs)
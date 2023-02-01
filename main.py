import time
import copy
import numpy as np
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from dataset import FocalLengthDataset

# original source: https://www.kaggle.com/code/ivankunyankin/resnet18-from-scratch-using-pytorch/notebook

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from resnet18 import ResNet_18

torch.manual_seed(17)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 256

# train set + train set with augmentations. factor describes random crop factor
fd_train = ConcatDataset([
    FocalLengthDataset(data_dir="./train"),
    FocalLengthDataset(data_dir="./train", factor=2)
])
fd_val = FocalLengthDataset(data_dir="./val")

train_loader = DataLoader(fd_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(fd_val, batch_size=batch_size, shuffle=False)

model = ResNet_18(3, 1)

model.to(device)
next(model.parameters()).is_cuda

# ######################################################################
# # TrainingLoop
# ######################################################################

epochs = 1000
criterion = nn.L1Loss()
lr=1e-4
o = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

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
                    loss = criterion(torch.log(outputs).squeeze(), torch.log(labels).squeeze())
                    
                    preds = torch.round(outputs)

                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
                
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            wandb.log({
                f"{phase}_accuracy": epoch_acc,
                f"{phase}_loss": epoch_loss
            })

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        if epoch % 100 == 0:
            torch.save(model, f"model-{epoch}.pt")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history

if __name__ == "__main__":
    wandb.init(project="focal-length-est", entity="wheresmycookie")

    model, _ = train_model(model, {"train": train_loader, "val": val_loader}, criterion, o, epochs)

    torch.save(model, "model-best.pt")

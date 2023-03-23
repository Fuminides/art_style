import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from torchvision import models
from semart_loader import ArtDatasetMTL
import parser

args_dict = parser.parser.parse_args()

model = models.resnet()
model_checkpoint = torch.load(args_dict.model_path)
model.load_state_dict(model_checkpoint['model_state_dict'])
print('Loaded model from checkpoint in epoch: ', model_checkpoint['epoch'])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
    
cluster_path = args_dict.semart_path


train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
    transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
    transforms.CenterCrop(224),                         # we get only the center of that rescaled
    transforms.ToTensor(),                              # to pytorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                            std=[0.229, 0.224, 0.225])
])

semart_train_loader = ArtDatasetMTL(args_dict, set='train', transform=train_transforms)
semart_val_loader = ArtDatasetMTL(args_dict, set='val', transform=val_transforms)
semart_test_loader = ArtDatasetMTL(args_dict, set='test', transform=val_transforms)

train_loader = torch.utils.data.DataLoader(
    semart_train_loader,
    batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
print('Training loader with %d samples' % semart_train_loader.__len__())

val_loader = torch.utils.data.DataLoader(
    semart_val_loader,
    batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
print('Validation loader with %d samples' % semart_val_loader.__len__())

test_loader = torch.utils.data.DataLoader(
    semart_test_loader,
    batch_size=args_dict.batch_size, shuffle=True, pin_memory=True, num_workers=args_dict.workers)
print('Validation loader with %d samples' % semart_val_loader.__len__())

# Set the model to evaluation mode
model.eval()

sets_name = ['train', 'val', 'test']
for ix, set_loader in enumerate([train_loader, val_loader, test_loader]):
    # Make predictions on the test set
    predictions = []
    with torch.no_grad():
        for data in set_loader:
            inputs, labels = data
            labels = labels.to(device)
            outputs = model(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            predictions.append(outputs.cpu().numpy())
    
    pd.DataFrame(predictions).to_csv(cluster_path + '/predictions_' + sets_name[ix] + '.csv', index=False)


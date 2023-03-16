import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wikiart_loader
import pandas as pd

import parser

args = parser.parser.parse_args()
cluster_path = args.data_path

train_path = cluster_path 
test_path = cluster_path 


df_styles_train = pd.read_csv(train_path + args.train_file, index_col=0)
df_styles_test = pd.read_csv(test_path + args.val_file, index_col=0)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device == "cpu":
    print("Training on CPU")
else:
    print("Training on GPU")

train_transforms = transforms.Compose([
        transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256),                         # we get only the center of that rescaled
        transforms.RandomCrop(224),                         # random crop within the center crop (data augmentation)
        transforms.RandomHorizontalFlip(),                  # random horizontal flip (data augmentation)
        transforms.ToTensor(),                              # to pytorch tensor
        transforms.Lambda(lambda x: x.to(device)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                             std=[0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
    transforms.Resize(256),                             # rescale the image keeping the original aspect ratio
    transforms.CenterCrop(224),                         # we get only the center of that rescaled
    transforms.ToTensor(),                              # to pytorch tensor
    transforms.Lambda(lambda x: x.to(device)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406, ],  # ImageNet mean substraction
                            std=[0.229, 0.224, 0.225])
])

train_data = wikiart_loader.WikiArtLoader(df_styles_train, train_path, transform=train_transforms)
test_data = wikiart_loader.WikiArtLoader(df_styles_test, test_path, transform=val_transforms)

train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

# Define the model, loss function and optimizer
from torchvision import models
resnet = models.resnet50(pretrained=True)
model = resnet
model = model.to(device)
    
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# Define the train function
def train(epochs):
    for epoch in range(epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        for i, data in enumerate(train_dataloader):
            inputs, labels = data
            labels = labels.to(device)
            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and update the weights
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            #if i % 10 == 0:    # print every 10 mini-batches
        print('Loss: %.3f' %
                (running_loss / len(train_dataloader)))
        running_loss = 0.0
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        }, args.model_path.split('.')[0] + '_checkpoint.pt')
    
    print('Finished Training')

# Define the test function
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    return correct / total

train(epochs=args.epochs)
accuracy = test()
print('Accuracy in test: ', accuracy)
torch.save(model.state_dict(), args.model_path)


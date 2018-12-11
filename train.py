import matplotlib.pyplot as plt
import numpy as np
import time

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import json

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("data_dir", help="directory of data")
parser.add_argument("--save_dir", help="save dir of checkpoint", default="./")
parser.add_argument("--arch", help="architecture", default="vgg13")
parser.add_argument("--learning_rate", help="learning rate", type=float, default=0.001)
parser.add_argument("--hidden_units", help="hidden units", type=int, default=512)
parser.add_argument("--epochs", help="epochs", type=int, default=20)
parser.add_argument("--gpu", help="use gpu", action="store_true")

args = parser.parse_args()

use_cuda = args.gpu

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# TODO: Define your transforms for the training, validation, and testing sets
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True)
               for x in ['train', 'valid', 'test']}


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# TODO: Build and train your network
class FFClassifier(nn.Module):
    
    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.5):
        super().__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        
        self.drop = nn.Dropout(p=drop_prob)
        
    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        
        x = F.log_softmax(x, dim=1)
        return x

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

# Create your own classifier
net = FFClassifier(25088, 4096, len(cat_to_name))

# Put your classifier on the pretrained network
model.classifier = net

# Define the loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

def validation(model, val_data, criterion, cuda=False):
    val_start = time.time()
    running_val_loss = 0
    accuracy = 0
    for inputs, labels in val_data:
        inputs, labels = Variable(inputs), Variable(labels)
        
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        val_loss = criterion(outputs, labels)

        ps = torch.exp(outputs.data)
        
        _, predicted = ps.max(dim=1)
        
        equals = predicted == labels.data
        accuracy += torch.sum(equals)/len(equals)

        running_val_loss += val_loss.data[0]
    val_time = time.time() - val_start
    print("Valid loss: {:.3f}".format(running_val_loss/len(dataloaders['valid'])),
          "Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])),
          "Val time: {:.3f} s/batch".format(val_time/len(dataloaders['valid'])))

epochs = args.epochs
print_every_n = 10

if use_cuda:
    model.cuda()
else:
    model.cpu()

model.train()
for e in range(epochs):
    print(f"Epoch {e+1}/{epochs}")
    counter = 0
    running_loss = 0
    for inputs, labels in dataloaders['train']:
        counter += 1
        
        # Training pass
        inputs, labels = Variable(inputs), Variable(labels)
        
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        optimizer.zero_grad()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        
        if counter % print_every_n == 0:
            print(f"Step: {counter}")
            print(f"Training loss {running_loss/counter:.3f}")
            model.eval()
            validation(model, dataloaders['valid'], criterion, cuda=use_cuda)
            model.train()
    else:
        # Validation pass
        train_end = time.time()
        model.eval()
        validation(model, dataloaders['valid'], criterion, cuda=use_cuda)

# TODO: Do validation on the test set
model.cpu()
model.eval()
validation(model, dataloaders['test'], criterion, cuda=use_cuda)

# TODO: Save the checkpoint 
model.class_to_idx\
= image_datasets['train'].class_to_idx
model.cpu()
torch.save({'arch': args.arch,
            'hidden': args.hidden_units,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}, 
            'classifier.pt')
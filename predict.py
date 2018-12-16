from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("input", help="input directory")
parser.add_argument("checkpoint", help="directory of data")
parser.add_argument("--top_k", help="topk", type=int, default=5)
parser.add_argument("--category_names", help="category to name file", default="cat_to_name.json")
parser.add_argument("--gpu", help="use gpu", action="store_true")

args = parser.parse_args()

use_cuda = args.gpu and torch.cuda.is_available()

data_dir = args.input
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

with open(args.category_names, 'r') as f:
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

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if checkpoint['arch'] == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif checkpoint['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Create the classifier
    net = FFClassifier(25088, checkpoint['hidden'], len(model.class_to_idx))

    # Put the classifier on the pretrained network
    model.classifier = net
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint(args.checkpoint) #load_checkpoint('classifier.pt')
if use_cuda:
    model.cuda()
else:
    model.cpu()
 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    aspect = image.size[0]/image.size[1]
    if aspect > 0:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left_margin = (image.width-224)/2
    top_margin = (image.height-224)/2
    image = image.crop((left_margin, top_margin, left_margin+224, top_margin+224))
    #normalize
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/std
    #move color channels
    image = image.transpose((2, 0, 1))
    
    return image

image_path = args.input + '/test/28/image_05230.jpg'
image = Image.open(image_path)
image = process_image(image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False)
    inputs = inputs.unsqueeze(0)
    ps = torch.exp(model.forward(inputs))
    top_probs, top_labels = ps.topk(topk)
    top_probs, top_labels = top_probs.data.numpy().squeeze(), top_labels.data.numpy().squeeze()
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[each] for each in top_labels]
    
    return top_probs, top_classes


# TODO: Display an image along with the top 5 classes
image_path = args.input + '/test/28/image_05230.jpg'

model.eval()
top_probs, top_classes = predict(image_path, model, args.top_k)
image = Image.open(image_path)
image = np.array(image)

fig, (img_ax, p_ax) = plt.subplots(figsize=(4,7), nrows=2)
img_ax.imshow(image)
img_ax.xaxis.set_visible(False)
img_ax.yaxis.set_visible(False)

p_ax.barh(np.arange(5, 0, -1), top_probs)

top_cat_names = [cat_to_name[each] for each in top_classes]

p_ax.set_yticks(range(1,6))
p_ax.set_yticklabels(reversed(top_cat_names))
fig.tight_layout(pad=0.1, h_pad=0)

dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}

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

# TODO: Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True)
               for x in ['train', 'valid', 'test']}

images, labels = next(iter(dataloaders['test']))

img_idx = 20

inputs = Variable(images[img_idx,:].unsqueeze(0))
model.eval()
# model.cpu()
ps = torch.exp(model.forward(inputs))

top_probs, top_classes = ps.topk(args.top_k)

idx_to_class = {val: key for key, val in model.class_to_idx.items()}
top_classes = [idx_to_class[each.item()] for each in top_classes.data.squeeze()]

fig, (img_ax, p_ax) = plt.subplots(figsize=(4,7), nrows=2)
img_ax = imshow(inputs.data.squeeze(), ax=img_ax)
img_ax.set_title(cat_to_name[idx_to_class[labels[img_idx].item()]])
img_ax.xaxis.set_visible(False)
img_ax.yaxis.set_visible(False)

p_ax.barh(np.arange(5, 0, -1), top_probs.data.numpy().squeeze())
top_cat_names = [cat_to_name[each] for each in top_classes]
p_ax.set_yticks(range(1,6))
p_ax.set_yticklabels(reversed(top_cat_names))
fig.tight_layout(pad=0.1, h_pad=0)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_root', type=str, default='/home/xywang/code/pruning/catdog_classification/train',
                    help='training dataset (default: train)')
parser.add_argument('--vaild_root', type=str, default='/home/xywang/code/pruning/catdog_classification/test',
                    help='training dataset (default: test)')
parser.add_argument('--sr',type=bool, default=True,
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--refine', default='', type=str, metavar='PATH',
                    help='path to the pruned model to be fine tuned')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 160)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save', default='./models', type=str, metavar='PATH',
                    help='path to save prune model (default: current directory)')
parser.add_argument('--arch', default='resnet', type=str, 
                    help='architecture to use')
parser.add_argument('--scratch',default='', type=str,
                    help='the PATH to the pruned model')

args = parser.parse_args()
device = torch.device('cuda:0')

if not os.path.exists(args.save):
    os.makedirs(args.save)

#数据加载及处理
train_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224,scale=(0.6,1.0),ratio=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224,scale=(1.0,1.0),ratio=(1.0,1.0)),
    # transforms.RandomResizedCrop(224,scale=(0.6,1.0),ratio=(0.8,1.0)),
    # transforms.RandomHorizontalFlip(),
    # torchvision.transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
    # torchvision.transforms.ColorJitter(brightness=0, contrast=0.5, saturation=0, hue=0),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

train_data =  torchvision.datasets.ImageFolder(
        root=args.train_root,
        transform=train_transform
    )

vaild_data = torchvision.datasets.ImageFolder(
        root=args.vaild_root,
        transform=test_transform
    )

train_set = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size,
    shuffle=True
)

test_set = torch.utils.data.DataLoader(
    vaild_data,
    batch_size=args.batch_size,
    shuffle=False
)

def updateBN(model, s ,pruning_modules):
    for module in pruning_modules:
        module.weight.grad.data.add_(s * torch.sign(module.weight.data))
#训练和验证
criteration = nn.CrossEntropyLoss()
def train(model,device,dataset,optimizer,epoch,pruning_modules):
    model.train()
    correct = 0
    for i,(x,y) in tqdm(enumerate(dataset)):
        x , y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss =  criteration(output,y)     
        loss.backward()
        optimizer.step()

        if args.sr:
            updateBN(model,args.s,pruning_modules)
        
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.3f}%)".format(epoch,loss,correct,len(dataset)*args.batch_size,100*correct/(len(dataset)*args.batch_size)))
    

def vaild(model,device,dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i,(x,y) in tqdm(enumerate(dataset)):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = criteration(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    print("Test Loss {:.4f} Accuracy {}/{} ({:.3f}%)".format(loss,correct,len(dataset)*args.batch_size,100*correct/(len(dataset)*args.batch_size)))

def get_pruning_modules(model):
    module_list = []
    for module in model.modules():
        if isinstance(module,torchvision.models.resnet.Bottleneck):
            module_list.append(module.bn1)
            module_list.append(module.bn2)
    return module_list

model = torchvision.models.resnet50(pretrained=True)
model.fc = nn.Sequential(
        nn.Linear(2048,2)
    )
model.to(device)

pruning_modules = get_pruning_modules(model)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(1,args.epochs + 1):
    train(model,device,train_set,optimizer,epoch,pruning_modules)
    vaild(model,device,test_set)
    torch.save(model.state_dict(), 'models/model_pruning.pth')



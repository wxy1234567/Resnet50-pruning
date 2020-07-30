import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import torch_pruning as tp
import time

#超参
device = torch.device('cuda:1')
LR = 0.001
EPOCH = 50
BTACH_SIZE = 100
train_root = '/home/xywang/code/pruning/catdog_classification/train'
vaild_root = '/home/xywang/code/pruning/catdog_classification/test'


#数据加载及处理
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

vaild_data = torchvision.datasets.ImageFolder(
        root=vaild_root,
        transform=test_transform
    )

test_set = torch.utils.data.DataLoader(
    vaild_data,
    batch_size=BTACH_SIZE,
    shuffle=False
)

def updateBN(model, s ,pruning_modules):
    for module in pruning_modules:
        module.weight.grad.data.add_(s * torch.sign(module.weight.data))
    
#训练和验证
criteration = nn.CrossEntropyLoss()

def vaild(model,device,dataset):
    model.eval().to(device)
    correct = 0
    with torch.no_grad():
        for i,(x,y) in tqdm(enumerate(dataset)):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = criteration(output,y)
            pred = output.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    print("Test Loss {:.4f} Accuracy {}/{} ({:.3f}%)".format(loss,correct,len(dataset)*BTACH_SIZE,100*correct/(len(dataset)*BTACH_SIZE)))
    return 100*correct/(len(dataset)*BTACH_SIZE)

def get_pruning_modules(model):
    data = []
    for layer in model.named_modules():
        if "bn" in layer[0]:
            data.append(layer[1])
    return data

model = torch.load('models/model_pruned_unknow.pth')
print(model)
prec2 = vaild(model,device,test_set)

random_input = torch.rand((16, 3, 224, 224)).to(device)
def test_speed(input, model, repeat=200):
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i in range(repeat):
            output = model(input)
    avg_infer_time = (time.time() - start) / repeat

    return avg_infer_time


model_original = torchvision.models.resnet50()
model_original.fc = nn.Sequential(
        nn.Linear(2048,2)
    )
model_original.to(device)
model_original.load_state_dict(torch.load('models/model_pruning.pth'))

obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

prec1 = vaild(model_original,device,test_set)
inference_time = test_speed(random_input, model_original, repeat=100)
parameters_num = obtain_num_parameters(model_original)

pruning_inference_time = test_speed(random_input, model, repeat=100)
pruning_parameters_num = obtain_num_parameters(model)

print("original accuracy: ", prec1)
print("original parameters_num: ",parameters_num)
print("original inference time: "+str(inference_time))

print("pruning accuracy: ", prec2)
print("pruning parameters_num: ",pruning_parameters_num)
print("pruning inference time: "+str(pruning_inference_time))

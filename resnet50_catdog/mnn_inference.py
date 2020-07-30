from __future__ import print_function
import numpy as np
import MNN
import cv2
import os, sys
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import torch

test_root = '/home/xywang/code/pruning/catdog_classification/test'
device = torch.device("cuda:0")
batch_size = 1
test_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomResizedCrop(224,scale=(1.0,1.0),ratio=(1.0,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

vaild_data = torchvision.datasets.ImageFolder(
        root=test_root,
        transform=test_transform
    )

test_set = torch.utils.data.DataLoader(
    vaild_data,
    batch_size=batch_size,
    shuffle=False
)

def test(test_dir, test_model):
    interpreter = MNN.Interpreter(test_model)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    num = 0
    right = 0
    for batch_num, (data, target) in tqdm(enumerate(test_set)):
        data, target = data.to(device), target.to(device)
        data=data.cuda().data.cpu().numpy()
        tmp_input = MNN.Tensor((1, 3, 224, 224), MNN.Halide_Type_Float,\
                               data, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        interpreter.runSession(session)
        output = interpreter.getSessionOutput(session)
        output = np.array(output.getData())
        output = torch.from_numpy(output).unsqueeze(0)
        prediction = torch.max(output,1)[1]
        num += target.size(0)
        #print(target,prediction)
        right += np.sum(prediction.cpu().numpy()== target.cpu().numpy())

        if num % 100 == 0:
            print('current num {} : {} '.format(num, right/num))

    print('All num {} : {} '.format(num, right/num))
    
if __name__ == "__main__":
    test_dir = '/home/xywang/code/pruning/catdog_classification/test'
    #test(test_dir, "models/model_pruned_unknow.mnn")
    test(test_dir, "models/model_pruned_unknow_quantized.mnn")
    #test(test_dir, "models/model_pruned_0.5_quantized.mnn")
# Resnet50-pruning
This work implements the pruning of resnet50 cat and dog classification model.
Pruning tool based on https://github.com/VainF/Torch-Pruning.

## environment
numpy
torch
torchvison
matplotlib
tqdm
mnn

## Quickstart
You may need to change your own path in the code.
### dataset
```bash
wget -c https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip
```
Split the dataset into train and test at a ratio of 0.1, both of which contain cat and dog.

cd resnet50_catdog
### Basic training:
```bash
python train.py --sr False
```
### Sparse training:
```bash
python train.py --sr True --s 0.0001
```
### Pruning and finetune:
```bash
python prune.py --percent 0.8 --sr False
```
### Test
```bash
python test.py
```
## mnn inference
### to onnx
```bash
python pth2onnx.py
```
### onnx2mnn
```bash
mnnconvert [-h] -f {TF,CAFFE,ONNX,TFLITE,MNN} --modelFile MODELFILE
                  [--prototxt PROTOTXT] --MNNModel MNNMODEL [--fp16 FP16]
```
### quantize
Prepare feature quantization image
```bash
mnnquant src_mnn dst_mnn config
```
### mnn inference
```bash
python mnn_inference.py
```

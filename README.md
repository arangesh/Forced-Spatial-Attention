# Forced Spatial Attention for Driver Foot Activity Classification

This is the Pytorch implementation for the training procedure described in [Forced Spatial Attention for Driver Foot Activity Classification](http://cvrr.ucsd.edu/publications/2019/FSAFAC.pdf).

## Installation
1) Clone this repository
2) Install Pipenv:
```shell
pip3 install pipenv
```
3) Install all requirements and dependencies in a new virtual environment using Pipenv:
```shell
cd Forced-Spatial-Attention
pipenv install
```
4) Get link for desired PyTorch and Torchvision wheel from [here](https://download.pytorch.org/whl/torch_stable.html) and install it in the Pipenv virtual environment as follows:
```shell
pipenv install https://download.pytorch.org/whl/cu100/torch-1.2.0-cp36-cp36m-manylinux1_x86_64.whl
pipenv install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl
```

## Dataset
1) Download the trainval dataset for driver foot activity classification using [this link]().
2) Extract the data.

## Training
The prescribed two-stage training procedure for the classification network can be carried out as follows:
```shell
pipenv shell # activate virtual environment
python train_stage1.py --dataset-root-path=/path/to/dataset/ --snapshot=./weights/squeezenet1_1_imagenet.pth --version=1_1 --FSA
python train_stage2.py --dataset-root-path=/path/to/dataset/ --snapshot=/path/to/snapshot/from/stage1/training --version=1_1 --FSA
exit # exit virtual environment
```

## Inference
Pretrained weights for SqueezeNet v1.1 using the two-stage FSA loss can be found [here](https://github.com/arangesh/Forced-Spatial-Attention/blob/master/weights/squeezenet_1_1.pth). Inference can be carried out using [this](https://github.com/arangesh/Forced-Spatial-Attention/blob/master/demo.py) script as follows:
```shell
pipenv shell # activate virtual environment
python demo.py --video=/path/to/dataset/foot.mp4 --snapshot=/path/to/snapshot --version=1_1
exit # exit virtual environment
```

Config files, logs, results and snapshots from running the above scripts will be stored in the `Forced-Spatial-Attention
/experiments` folder by default.

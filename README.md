# underwater-dense-depth-completion

This repo is the PyTorch implementation of our work  ["Fast Monocular Depth Estimation for Autonomous Underwater Vehicles"](), developed by Reut Azaria and Guy Gilboa at Technion Institute of Technology.
<p align="center">
	<img src="https://raw.githubusercontent.com/reutazaria/underwater_dense_depth_completion/master/movie_depth_Tukey_var.gif" alt="photo not available" height="100%" width="100%">
</p>

Our network is trained with the "Nachsholim" dataset alone. The use of additional data is likely to further improve the accuracy.

## Contents
1. [Dependency](#dependency)
0. [Data](#data)
0. [Trained Models](#trained-models)
0. [Commands](#commands)
<!--- 0. [Citation](#citation) --->


## Dependency
This code was tested with Python 3 and PyTorch 1.0 on Ubuntu 16.04. 
For installing the environment download `SelfDepth.yml` file and run the following command:
```bash
conda env create -f SelfDepth.yml
```

## Data
The datasets will be stored in the `../data/` folder, divided into sub-directories for every single dataset. 
Each dataset folder should be divided into gt, rgb and sparse sub-folders containing ground truth depth maps, RGB images, and sparse maps, respectively.
Each one of the laters should be divided into a test, train, and validation set.

The overall code, data, and result directories are structured as follows:
```
.
├── underwater-dense-depth-completion
├── data
|   ├── Nachsholim
|   |   ├── gt
|   |   |   ├── test
|   |   |   ├── train
|   |   |   ├── val
|   |   ├── rgb
|   |   |   ├── test
|   |   |   ├── train
|   |   |   ├── val
|   |   ├── sparse
|   |   |   ├── test
|   |   |   ├── train
|   |   |   ├── val
|   ├── dataset#2
|   ├── dataset#3
|   |  
|   |   
├── results
```
Each dataset has its corresponding data loader file, for example `nachsholim_loader.py`.
This file specifies the relevant paths, as well as the input image size 
(since the model is based on a CNN architecture with 5 encoding/decoding layers, the image height and width should be divisible by 32). 

## Trained Models
<!---
Download our trained models at http://datasets.lids.mit.edu/self-supervised-depth-completion to a folder of your choice.
- supervised training (i.e., models trained with semi-dense lidar ground truth): http://datasets.lids.mit.edu/self-supervised-depth-completion/supervised/
- self-supervised (i.e., photometric loss + sparse depth loss + smoothness loss): http://datasets.lids.mit.edu/self-supervised-depth-completion/self-supervised/
--->

## Commands
A complete list of training options is available with 
```bash
python main.py -h
```
For instance,
<!---
```bash
# train with the KITTI semi-dense annotations, rgbd input, and batch size 1
python main.py --train-mode dense -b 1 --input rgbd

# train with the self-supervised framework without using ground truth
python main.py --train-mode sparse 

# resume previous training
python main.py --resume [checkpoint-path] 

# test the trained model on the test set
python main.py --evaluate [checkpoint-path] --val full
```
--->
<!--- ## Citation --->
 <!--- If you use our code or method in your work, please cite the following: --->


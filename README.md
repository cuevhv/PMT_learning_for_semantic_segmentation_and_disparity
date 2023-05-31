# README #

Modification of https://bitbucket.org/cuevhv/jointsegdispnet/src/master/ to use it with S-ROSeS dataset on pytorch.

### Dependencies ###
- Pip dependencies: ipython, Ipython, opencv-python, opencv-contrib-python, scikit-learn, tqdm, scikit-image, matplotlib, imgaug, pandas, tensorboard, seaborn, openexr, h5py
- Torch dependencies: spatial-correlation-sampler, efficientnet_pytorch 
- Tested in:
  - Ubuntu 20.04
  - Python 3.7
  - Torch 1.14.0
  - Cuda 11.8.0 
  - cuDNN 8.7.0.84

### How it works? ###
The network uses a correlation layer.

Tested correlation1d layers (the checked one is currently being used):

- [ ] https://github.com/luoru/correlation1d-tf - It works with a simple example but it has some problems with the dimensions when training it.
- [ ] https://github.com/fedor-chervinskii/dispflownet-tf - Could not compile. 
- [X] https://github.com/tensorflow/addons - It is for optical flow rather than for disparity.

To run:
`python torch_implementation.py -colorL ${.txt with list of left color images} -colorR ${.txt with list of right color images} -seg ${.txt with list of left segmented images} -disp ${.txt with list of left disparities}`


### Execution on Docker
There is an init script to initialize the container `scripts/scriptsDocker/initDocker.sh`.

### Dataset "ROSeS_depth_4_types"
Dataset with color images, segmentation and disparity ready to be trained available at `https://u.pcloud.link/publink/show?code=XZFfEWVZaqLvM2EBwg4HWLo4vyJ5x8RTkhDy`
Should be placed on the root folder.

### Train, validation and test set 
Execute `python scripts/divideLeftRightTrainVal.py` to split the dataset into train, val and tests sets.
Execute `python reduceExistentDataset.py` to create a subset of the existent ones, ideal to realize tests with less computation requirements.

### Train or eval the model
Script to train the model `scripts/trainTorchImpl.sh`.
Script to evaluate the model and obtain metrics `scripts/evalTorchImpl.sh`.

### Disparity images 
Obtain disparity files from dispersion ".exr" files with `python scripts/obtainDispFromDepth.py`.

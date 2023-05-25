# Progressive Multi-task Joint Learning for Semantic Segmentation and Disparity Estimation

Modification of https://bitbucket.org/cuevhv/jointsegdispnet/src/master/ to use it with ROSeS dataset on pytorch.

## Docker
There is an init script at "scripts/scriptsDocker" to initialize the container

## Dataset "ROSeS_depth_4_types"
Dataset with color images, segmentation and disparity ready to be trained available at https://u.pcloud.link/publink/show?code=XZFfEWVZaqLvM2EBwg4HWLo4vyJ5x8RTkhDy
Should be placed on root folder of "jointsegdispnet"

## Train, validation and test set 
Execute "divideLeftRightTrainVal.py" to split the dataset into train, val and tests sets.
Execute "reduceExistentDataset.py" to create a subset of the existent ones, ideal to realize tests with less computation requirements.

## Train or eval the model
"trainTorchImpl.py" and "EvalTorchImpl.py" in "scripts" folder.

## Disparity images 
Obtain disparity files from dispersion ".exr" files with "scripts/obtainDispFromDepth.py"

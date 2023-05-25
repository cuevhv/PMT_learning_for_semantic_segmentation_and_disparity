#!/bin/bash

echo "Execute 'tf' or 'torch'. Type jupyter to execute jupyter"

num_gpu="6"

# Create image
if [[ $1 == "tf" ]]
then
  name_im="rosales-gpu-tf"
  echo "TensorFlow $name_im"
  docker build -t $name_im TF
else
  name_im="rosales-gpu-torch"
  echo "PyTorch $name_im"
  docker build -t $name_im Torch --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g)
fi




if [[ $2 == "jupyter" ]]
then
  docker run -p 8889:8888 -it --name $name_im -v /media/nomo/linuxbackup/Rosales:/Rosales --rm --gpus all $name_im jupyter-lab --ip 0.0.0.0 --port 8888 --allow-root #-e DISPLAY=$DISPLAY
else
  docker run  -it -u $(id -u):$(id -g) --name $name_im -v $(pwd)/../../:/home/user/jointsegdispnet --rm --gpus "device=${num_gpu}" $name_im /bin/bash 
fi



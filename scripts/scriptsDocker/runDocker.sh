#!/bin/bash
# Just run docker container already created on "initDocker"
name_im="rosales-gpu-torch"
docker run  -it -u $(id -u):$(id -g) --name $name_im -v $(pwd)/../../:/home/user/jointsegdispnet --rm --gpus "device=1" $name_im /bin/bash 

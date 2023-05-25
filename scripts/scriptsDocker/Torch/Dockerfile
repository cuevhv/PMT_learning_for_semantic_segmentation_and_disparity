# Prepared to execute "jointsegdispnet" pytorch version. 

FROM nvcr.io/nvidia/pytorch:22.12-py3
ENV DEBIAN_FRONTEND noninteractive


####### Allow rootless acess
RUN apt-get -y update \
&& DEBIAN_FRONTEND=noninteractive apt-get -y install sudo curl locales \ 
&& rm -rf /var/lib/apt/lists/*
RUN apt-get -y upgrade

RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8   
ENV LANGUAGE en_US:en ENV LC_ALL en_US.UTF-8

# Parameters on the local machine 
ARG USER_ID=10008
ARG GROUP_ID=10001

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user

ENV HOME=/home/user
RUN chmod 777 /home/user
ENV PATH="/home/user/.local/bin:${PATH}"
WORKDIR /home/user


RUN mkdir /home/user/jointsegdispnet/
RUN mkdir /home/user/jointsegdispnet/scripts


ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



RUN apt-get -y update 
# Pillow
RUN apt-get install -y libjpeg-dev zlib1g-dev

# General dependencies
RUN python3 -m pip install --upgrade pip
RUN pip3 install  ipython Ipython opencv-python opencv-contrib-python 
RUN pip3 install  -U scikit-learn
RUN pip3 install  tqdm scikit-image matplotlib imgaug
RUN pip3 install  pandas seaborn tensorboard
# Torch dependencies
RUN pip3 install  -U git+https://github.com/szagoruyko/pytorchviz
RUN pip3 install  git+https://github.com/frgfm/pytorch-summary@densenet-fix
RUN pip3 install  spatial-correlation-sampler efficientnet_pytorch

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg libsm6 libxext6 

RUN pip3 install jupyterlab tabulate 
WORKDIR /home/user/jointsegdispnet/scripts

RUN apt-get install libopenexr-dev
RUN pip3 install openexr h5py

#RUN git clone https://github.com/NVIDIA/apex
#RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" apex/


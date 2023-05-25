FROM tensorflow/tensorflow:latest-gpu

RUN python3 -m pip install --upgrade pip
RUN apt update
RUN pip3 install --upgrade numpy==1.20.3
RUN pip install matplotlib scipy theano
RUN apt install -y python3-opencv libopenexr-dev
RUN pip install scikit-learn scikit-image openexr

# Test pfm
# python py_stereo_pfm.py -left ../jointsegdispnet/ROSeS_depth_4_types/val_list_of_left_images.txt -right ../jointsegdispnet/ROSeS_depth_4_types/val_list_of_right_images.txt -pfm ../jointsegdispnet/ROSeS_depth_4_types/val_list_of_left_depth_images.txt

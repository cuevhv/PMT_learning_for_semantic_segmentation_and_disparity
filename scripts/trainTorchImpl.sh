#!/bin/bash

# Script to train the method with the "ROSeS_depth_4_types" dataset

cd ..

# Relevant args
batch_size=8
epochs_number=100
datasetFolder="ROSeS_depth_4_types_ORIGINAL"
log_file="log/log_train_data_ORIGINAL.txt"

# Dataset small or big
#typeDataset="small"
typeDataset=""

# Big dataset
folder=""
sufix=""

if [[ $typeDataset == "small" ]]
then
    # Small dataset
    folder="reduced_size/"
    sufix="__small"
fi

rm ${log_file} 

python3 torch_implementation.py -gpu_n 0 -save_img 0 -n 1 -f16 0 -torch_amp 0 -acmt_grad 1 -freeze_bn 0 \
        -convDeconvOut 0 -train 1 -only_test 0 -crop 256 512 -corrType 1dcorr \
        -net sdnet_mini_ext -optimType adam -backbone densenet -datasetName roses -n_data 10 -b $batch_size -e $epochs_number \
        -dropout 0.0 -edges 0 -hanet 0 -use_att 1 -multaskloss 0 -aspp 0 -loss cross_entropy lovasz_loss \
        -segWeight 1 -output_activation linear \
        -colorL ${datasetFolder}/${folder}train_list_of_left_images${sufix}.txt \
        -colorR ${datasetFolder}/${folder}train_list_of_right_images${sufix}.txt \
        -seg ${datasetFolder}/${folder}train_list_of_left_seg_gt${sufix}.txt \
        -inst ${datasetFolder}/${folder}train_inst_all${sufix}.txt \
        -disp ${datasetFolder}/${folder}train_list_of_left_disp_images${sufix}.txt \
        -colorL_test ${datasetFolder}/${folder}val_list_of_left_images${sufix}.txt \
        -colorR_test ${datasetFolder}/${folder}val_list_of_right_images${sufix}.txt \
        -seg_test ${datasetFolder}/${folder}val_list_of_left_seg_gt${sufix}.txt \
        -disp_test ${datasetFolder}/${folder}val_list_of_left_disp_images${sufix}.txt \
        -w_savePath $(pwd) -show_results 0 -abilation None > ${log_file} 
	


#!/bin/bash

# Script to eval the method trained with custom weights on the "ROSeS_depth_4_types" dataset

cd ..

rm testResults/*.jpg

# Arguments
datasetFolder="ROSeS_depth_4_types_ORIGINAL"
weights="model_sdnet_mini_ext_i256_512_e100_b8_alinear_osmallOutSeg_w1_lcross_entropy_lovasz_loss_cr1dcorr_aspp0_optimadam_backbonedensenet_abltNone_att1_dropout0.0_dataroses_model_best_IOU0.9817_Derr0.0001.pth.tar"


# Dataset small or big
# typeDataset="small"
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


python3 torch_implementation.py -gpu_n 0 -train 0 -save_img 0 -n 1 -f16 0 -torch_amp 0 -acmt_grad 1 -freeze_bn 0 \
        -convDeconvOut 0 -only_test 0 -crop 256 512 -corrType 1dcorr \
        -net sdnet_mini_ext -optimType adam -backbone densenet -datasetName roses -n_data 10 -b 1 -e 10 \
        -dropout 0.0 -edges 0 -hanet 0 -use_att 1 -multaskloss 0 -aspp 0 -loss cross_entropy lovasz_loss \
        -segWeight 1 -output_activation linear \
        -colorL_test ${datasetFolder}/${folder}val_list_of_left_images${sufix}.txt \
        -colorR_test ${datasetFolder}/${folder}val_list_of_right_images${sufix}.txt \
        -seg_test ${datasetFolder}/${folder}val_list_of_left_seg_gt${sufix}.txt \
        -disp_test ${datasetFolder}/${folder}val_list_of_left_disp_images${sufix}.txt \
        -w_savePath $(pwd) -show_results 1 -abilation None \
        -load_weights weights/$weights





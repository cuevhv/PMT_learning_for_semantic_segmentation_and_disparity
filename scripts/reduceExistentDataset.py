#!/usr/bin/python

# Created to be executed on "ROSeS_depth_4_types" dataset
# Duplication of text files, with a smaller version to make tests. Must be executed after "divideLeftRight.py"

import os
os.chdir("../ROSeS_depth_4_types") # Executed from scripts folder

# Limit of files of the new small datasets
# limitFilesTrain, limitFilesVal = 1600, 400 
# limitFilesTrain, limitFilesVal = 400, 100 
# limitFilesTrain, limitFilesVal, limitFilesTest = 2133, 400, 20
limitFilesTrain, limitFilesVal, limitFilesTest = 100, 50, 2
folderName = "reduced_size/"

# Generic read and replicate
def readReplicateFile(fileName):
    f = open(fileName, "r")
    lines = f.readlines()
    f.close()    

    newFileLines = ""
    
    limitFiles = limitFilesTrain
    if fileName.startswith("val"):
        limitFiles = limitFilesVal
    elif fileName.startswith("test"):
        limitFiles = limitFilesTest

    for counterLines, l in enumerate(lines):
        if counterLines >= limitFiles:
            break
        # It is being introduced in a subfolder
        newFileLines += "../" + l 
        counterLines += 1
    
    outputFile = open(folderName + fileName[:-4] + "__small.txt", "w")
    outputFile.writelines(newFileLines)
    outputFile.close()

# Left images
left_images = ["train_list_of_left_images.txt", "train_list_of_left_seg_gt.txt", "train_list_of_left_depth_images.txt",
               "val_list_of_left_images.txt", "val_list_of_left_seg_gt.txt", "val_list_of_left_depth_images.txt",
               "test_list_of_left_images.txt", "test_list_of_left_seg_gt.txt", "test_list_of_left_depth_images.txt"]

for file_txt in left_images:
    readReplicateFile(file_txt)

# Right images
right_images = ["train_list_of_right_images.txt", "train_list_of_right_seg_gt.txt", 
                "val_list_of_right_images.txt", "val_list_of_right_seg_gt.txt",
                "test_list_of_right_images.txt", "test_list_of_right_seg_gt.txt",]


for file_txt in right_images:
    readReplicateFile(file_txt)


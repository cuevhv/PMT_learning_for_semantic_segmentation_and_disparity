#!/usr/bin/python

# Created to be executed on "ROSeS_depth_4_types" dataset, after "divideLeftRightTrainVal.py"

import os, random
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


np.random.seed(seed=0)
os.chdir("../ROSeS_depth_4_types_ORIGINAL/") # Executed from scripts folder

# Remove previous train/val files
os.system("rm train_* val_*")

f_mains = ["list_of_left_and_right_images.txt", "list_of_left_and_right_seg_images.txt"]
prefix, sufix = "list_of_", ["_images.txt", "_seg_gt.txt", "_disp.txt"]
folder = ""
percTrain, percVal, percTest = 0.8, 0.15, 0.05 # 80% training, 20% test.


def divide_dataset(dict_files):
    assert len(dict_files) > 0
    dict_train_files = {}
    dict_val_files = {}

    dict_files['img_l'], dict_files['img_r'], dict_files['seg_l'],\
    dict_files['seg_r'] = shuffle(dict_files['img_l'], dict_files['img_r'],\
                                                      dict_files['seg_l'], dict_files['seg_r'], random_state=0)
                                                      #dict_files['disp'], random_state=0)

    dict_train_files['img_l'], dict_val_files['img_l'], dict_train_files['seg_l'], dict_val_files['seg_l'] = train_test_split(dict_files['img_l'], dict_files['seg_l'], test_size = 1 - percTrain, shuffle=True, random_state=42)
    dict_train_files['img_r'], dict_val_files['img_r'], dict_train_files['seg_r'], dict_val_files['seg_r'] = train_test_split(dict_files['img_r'], dict_files['seg_r'], test_size = 1 - percTrain, shuffle=True, random_state=42)   

    return dict_train_files, dict_val_files

def saveFile (key_name, lines):    
    outputFile = open(key_name, "w")
    if not isinstance(lines, str) and not isinstance(lines, list):
        outputFile.writelines(lines.tolist())
    else:
        outputFile.writelines(lines)
    outputFile.close()

def readFile (nameFile):
    f_images = open(folder + nameFile, "r")
    lines = f_images.readlines()
    f_images.close()
    return lines

distributions = {}

folders = ["list_of_left_images.txt", "list_of_right_images.txt", "list_of_left_seg_images.txt", "list_of_right_seg_images.txt"] #, "list_of_disparity_outputs.txt"]
images_left, images_right = readFile(folders[0]), readFile(folders[1])
seg_left, seg_right = readFile(folders[2]), readFile(folders[3])

dict_files = {'img_l': images_left, 'img_r': images_right, 
              'seg_l': seg_left, 'seg_r': seg_right} #, 'disp': disp}
dict_train_files, dict_val_files = divide_dataset(dict_files)

dict_files_t = ['img', 'seg'] #, 'disp']
for n, file in enumerate(dict_files_t):
    if file == 'disp':
        continue # Extract disp from other function
        output_files = [["train_", "left", dict_train_files['disp']], ["val_", "left", dict_val_files['disp']]]
    else:
        left_train, right_train = dict_train_files[file + '_l'], dict_train_files[file + '_r']
        left_val, right_val     = dict_val_files[file + '_l'], dict_val_files[file + '_r']
        
        output_files = [["train_", "left", left_train], ["val_", "left", left_val],
                        ["train_", "right", right_train], ["val_", "right", right_val]]
    for o_f in output_files:
        saveFile(o_f[0] + prefix + o_f[1] + sufix[n], o_f[2])    


# disp has other names, can be extracted directly form left image
def saveDispFromImages(train_or_val):
    disp = ""
    # After that we create the location of disp files duplicating locations of others
    fileDispName = train_or_val + prefix + "left_disp" + sufix[0]
    fileLeft = train_or_val + prefix + "left" + sufix[1]
    os.system("cp " + fileLeft + " " + fileDispName)
    fl = open(fileLeft, "r")
    linesl = fl.readlines() 
    fl.close()

    for l in linesl:
        l = l.replace("seg", "disp")
        #disp += l[:-5] + ".pfm\n"
        disp += l[:-5] + ".png\n"
        
          
    saveFile(fileDispName, disp)
    
saveDispFromImages("train_")
saveDispFromImages("val_")



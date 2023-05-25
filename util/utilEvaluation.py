import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
import gc
from util.utilPlot import plot_learning_curves, gtPlot, cvPlot, display_data
import util.utilCityscape as cityscapes 
import time
import shutil
import os
#----------------------------------------------------------------------------------------------------------------------------------------------------
def unnormalize_disp(norm_disp, args):
    if len(norm_disp.shape) == 4:
        indx = 2
    else:
        indx = 1
    if args.output_activation == 'tanh':
        max_d = norm_disp.shape[indx]/2.0
        return np.where(norm_disp != -10, (norm_disp + 10) * float(max_d) / 20.0, 0)
    if args.output_activation == 'sigmoid':
        max_d = norm_disp.shape[indx]/2.0
        if args.dataset == 'garden':
            max_d = 25.0
        if args.dataset == 'kitti':
            max_d = 100.0
        return norm_disp*max_d#max_d 
    else:
        return norm_disp
#----------------------------------------------------------------------------------------------------------------------------------------------------

# def evaluatePrediction(X_test, y_gt, model, args, n_disp, n_seg):
#     if not os.path.isdir('results'):
#         os.mkdir('results')
#     if not os.path.isdir('results/seg'):
#         os.mkdir('results/seg')
#     if not os.path.isdir('results/disp'):
#         os.mkdir('results/disp')
    
#     y_pred = model.predict(X_test, batch_size=4)
#     res_seg, res_disp = 0, 0
#     save_result = True
#     title = ['Input Image 1', 'Input Image 2', 'disparity_original', 'disparity pred', 'segmentation original', 'segment pred']
#     for i in tqdm(range(y_gt[0].shape[0])):
#         disp_pred = unnormalize_disp(y_pred[n_disp][i], args)
#         disp_gt = unnormalize_disp(y_gt[0][i], args)
    
#         seg_pred = np.argmax(y_pred[n_seg][i], -1)
#         seg_gt = np.argmax(y_gt[1][i], -1)

#         res_seg += np.sum(np.equal(seg_gt, seg_pred))
#         res_disp += np.sum(np.abs(disp_pred - disp_gt))
        
#         if save_result:
#             data_list = [tf.keras.preprocessing.image.array_to_img(np.squeeze(X_test[0][i])),
#                         tf.keras.preprocessing.image.array_to_img(np.squeeze(X_test[1][i])),
#                         np.squeeze(disp_gt), np.squeeze(disp_pred),
#                         np.squeeze(seg_gt) , np.squeeze(seg_pred)]
#             display_data(data_list, title)
            
#             plt.savefig('results/{}.png'.format(i))
#             plt.close()

#             seg_pred = Image.fromarray(np.squeeze(seg_pred).astype('uint8'), mode='L')
#             seg_pred.save(os.path.join('results/seg', str(i) + '.png'))
#             seg_pred = Image.fromarray(np.squeeze(disp_pred), mode='F')
#             seg_pred.save(os.path.join('results/disp', str(i) + '.png'))

#         # del disp_pred, disp_gt, seg_pred, seg_gt, data_list
#         # gc.collect()
        

#     dim = y_gt[n_disp].shape
#     print ('MAE disparity: ', res_disp/float(dim[0] * dim[1] * dim[2]))
#     print ('accuracy segmentation: ', res_seg/float(dim[0] * dim[1] * dim[2]))

def evaluatePredictionSequence(testing, model, args, Y_test_names=None, save_result=False):
    save_dir_names = ['big_res', 'small_res']
    if not os.path.isdir('results'):
        os.mkdir('results')
    for j in save_dir_names:
        if not os.path.isdir('results/'+j):
                os.mkdir('results/'+j)
        for i in ['seg', 'disp']:
            if not os.path.isdir('results/'+j+'/'+i):
                os.mkdir('results/'+j+'/'+i)
        

    res_seg, res_disp, res_d1_fg, res_d1_bg = 0, 0, 0, 0
    res_seg2, res_disp2, res_d1_fg2, res_d1_bg2 = 0, 0, 0, 0
    count = 0
    if args.dataset == 'kitti':
        label = 19#8
    IoU_count_vector = np.zeros(label)
    IoU_count_vector2 = np.zeros(label)
    IoU_vector = np.zeros(label)
    IoU_vector2 = np.zeros(label)

    for values in testing:
        if len(values[1]) > 2:
            pass
        for i in range(values[0]['left'].shape[0]):
            left = np.expand_dims(values[0]['left'][i], axis=0)
            right = np.expand_dims(values[0]['right'][i], axis=0)
            disp = np.expand_dims(values[1]['disp_out'][i], axis=0)
            seg = np.expand_dims(values[1]['seg_out'][i], axis=0)

            y_pred = model.predict([left, right])
            if Y_test_names == None:
                save_name = str(count)
            else:
                print(Y_test_names[count][0])
                save_name = Y_test_names[count][0].rsplit('/')[-1].rsplit('.')[0]

            seg_error, disp_error, d1_fg_error, IoU_error, IoU_count = get_diffSegDisp(disp, y_pred[0], seg, y_pred[-1], left, 
                                                                            right, args, save_name, save_result,
                                                                            save_dir = save_dir_names[0])
            seg_error2, disp_error2, d1_fg_error2, IoU_error2, IoU_count2 = get_diffSegDisp(disp, y_pred[1], seg, y_pred[-2], left,
                                                                                right, args, save_name, save_result,
                                                                                save_dir = save_dir_names[1])
            res_seg += seg_error
            res_disp += disp_error
            res_d1_fg += d1_fg_error
            IoU_vector += IoU_error
            IoU_count_vector+= IoU_count

            res_seg2 += seg_error2
            res_disp2 += disp_error2
            res_d1_fg2 += d1_fg_error2
            IoU_vector2 += IoU_error2
            IoU_count_vector2 += IoU_count2

            count+=1
            dim = disp.shape
            print('samples: ', count)
            print ('MAE disparity: ', res_disp/float(count * dim[1] * dim[2]))
            print ('accuracy segmentation: ', res_seg/float(count * dim[1] * dim[2]))
            print ('d1-fg: ', res_d1_fg/float(count * dim[1] * dim[2]))
            #print(['category {}: {}'.format(cityscapes.catid2category[i], IoU_vector[i]/IoU_count_vector[i]) for i in range(len(IoU_vector))]) 
            print('IoU per class (no void class): ', np.sum(IoU_vector[1:])/np.sum(IoU_count_vector[1:]))
            

            print ('MAE disparity 2: ', res_disp2/float(count * dim[1] * dim[2]))
            print ('accuracy segmentation 2: ', res_seg2/float(count * dim[1] * dim[2]))
            print ('d1-fg 2: ', res_d1_fg2/float(count * dim[1] * dim[2]))
            #print(['category {}: {}'.format(cityscapes.catid2category[i], IoU_vector2[i]/IoU_count_vector2[i]) for i in range(len(IoU_vector))]) 
            print('IoU per class2 (no void class): ', np.sum(IoU_vector2[1:])/np.sum(IoU_count_vector2[1:]))
            print('\n')
            
    #f= open("weights/{}.txt".format(args.load_weights.split('/')[-1]),"w+")
    old_best_d1 = 1000
    file_name = args.load_weights.rsplit('.', 1)
    print(file_name, 'split')
    score_text = "{}.txt".format(args.load_weights)
    if os.path.exists(score_text):
        with open(score_text, "r") as text_file:
            old_best_d1 = float(text_file.readline()[:-1])
    print('old best_d1', old_best_d1)

    best_d = res_d1_fg/float(count * dim[1] * dim[2]) + (1-res_seg/float(count * dim[1] * dim[2]))
    if best_d < old_best_d1:
        with open(score_text,"w") as text_file:
            best_d = res_d1_fg/float(count * dim[1] * dim[2]) + (1-res_seg/float(count * dim[1] * dim[2]))
            print('best d', best_d)
            best_d_string = "%.6f" % best_d
            text_file.write(best_d_string)
            copy_weights = file_name[0]+"_"+best_d_string+"."+file_name[1]
            shutil.copyfile(args.load_weights, copy_weights)
            print('weigths saved in: ', copy_weights)
    else:
        print('results did not improve')
    
  
    
def get_diffSegDisp(disp, disp_p, seg, seg_p, left, right, args, count=0, save_result=False, save_dir='results'):
    disp_pred = unnormalize_disp(disp_p, args)
    disp_gt = unnormalize_disp(disp, args)
    seg_pred = np.argmax(seg_p, -1)
    seg_gt = np.argmax(seg, -1)
    res_seg = np.sum(np.equal(seg_gt, seg_pred))
    
    abs_disp = np.abs(disp_pred - disp_gt)
    res_disp = np.sum(abs_disp)
    valid_disp = (disp_gt > 0.0)

    res_d1_fg = np.sum((abs_disp * valid_disp) > 3)
    
    IoU_vector, count_vector = segmentationIOU(np.squeeze(seg_gt), np.squeeze(seg_pred), args)

    if save_result:
        seg_pred = np.squeeze(seg_pred)
        disp_pred = np.squeeze(disp_pred)
        if args.output_type != 'seg_only':
            title = ['Input Image 1', 'Input Image 2', 'disparity_original', 'disparity pred', 'segmentation original', 'segment pred']
            data_list = [tf.keras.preprocessing.image.array_to_img(np.squeeze(left)),
                        tf.keras.preprocessing.image.array_to_img(np.squeeze(right)),
                        np.squeeze(disp_gt), disp_pred,
                        np.squeeze(seg_gt) , seg_pred]
        else:
            title = ['Input Image 1', 'Input Image 2', 'segmentation original', 'segment pred']
            data_list = [tf.keras.preprocessing.image.array_to_img(np.squeeze(left)),
                        tf.keras.preprocessing.image.array_to_img(np.squeeze(right)),
                        np.squeeze(seg_gt) , seg_pred]

        display_data(data_list, title)
        all_result_name = 'results/{}/{}.png'.format(save_dir, count)
        plt.savefig(all_result_name, dpi=200)
        plt.close()

        classSeg = np.zeros_like(seg_pred)
        values = np.unique(seg_pred.reshape(-1), axis=0)
        for i in values:
            mask = (seg_pred == i) * cityscapes.trainId2label[i].id
            classSeg += mask

        seg_name  = 'results/{}/seg/{}.png'.format(save_dir, str(count))
        disp_name = 'results/{}/disp/{}.png'.format(save_dir, str(count))

        classSeg = Image.fromarray(classSeg.astype('uint8'), mode='L')
        classSeg.save(seg_name)
        disp_pred = Image.fromarray(disp_pred, mode='F')
        if disp_pred.mode != 'RGB':
            disp_pred = disp_pred.convert('RGB')
        disp_pred.save(disp_name)

    return res_seg, res_disp, res_d1_fg, IoU_vector, count_vector

def segmentationIOU(gt, pred, args):
    th = 0.5
    if args.dataset == 'kitti':
        label = 19#8
    else:
        label = 9

    count_vector = np.zeros(label)
    IoU_vector = np.zeros(label)
    for i in range(label):
        gt_lbl = (gt == i) * 1
        IoU = -1
        if np.sum(gt_lbl) != 0:
            pred_label = (pred == i) * 1
            TP = gt_lbl * pred_label 
            FP = pred_label - TP
            FN = gt_lbl - TP
            
            img = np.concatenate((np.expand_dims(FN, -1), np.expand_dims(TP, -1), np.expand_dims(FP, -1)), axis=-1)
            
            # plt.imshow((img*255).astype(np.uint8))
            # plt.savefig('results/label_{}.png'.format(i), dpi=500)
            
            IoU = (np.sum(TP) / float(np.sum(TP) + np.sum(FP) + np.sum(FN)))
        IoU_vector[i] = IoU
        #print(cityscapes.catid2category[i], ': ', IoU)
    
    count_vector = (IoU_vector >= 0) * 1
    IoU_vector[IoU_vector == -1] = 0
    #IoU_vector = (IoU_vector >= 0.5) * 1
    return IoU_vector, count_vector

import numpy as np
from scipy.signal import find_peaks
import cv2
from utils.utilts_custom_class import * 
from scipy.ndimage import median_filter as med_filt
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

import torch
import torch.nn.functional as F
from numpy import ndarray as NDArray
from skimage import measure
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm import tqdm
from statistics import mean
import random
import os

def value_indx(list_for_index,value_find):
    indices = [i for i, x in enumerate(list_for_index) if x == value_find]
    return indices

def bbox_size_v1(diff_col_var,diff_row_var,col_indx,row_indx,no_of_inc_i = 30,no_of_inc_f=30):

    # Third quartile (Q3)

    #col_ini, col_end       = ini_final_indx(col_indx,no_of_val_inc= no_of_inc)
    #row_ini, row_end       = ini_final_indx(row_indx,no_of_val_inc= no_of_inc)
    col_ini, col_end        = 0, len(diff_col_var)#no_of_inc_i,len(diff_col_var)-no_of_inc_f
    row_ini, row_end        = 0, len(diff_row_var)#no_of_inc_i,len(diff_row_var)-no_of_inc_f

    Q3_c = np.percentile(diff_col_var[col_ini:col_end], 60, interpolation = 'midpoint') # best 40
        # Third quartile (Q3)
    Q3_r = np.percentile(diff_row_var[row_ini:row_end], 60, interpolation = 'midpoint')

    q3_c_var_val = np.array(diff_col_var>Q3_c,dtype='uint8')
    q3_r_var_val = np.array(diff_row_var>Q3_r,dtype='uint8')

    val = 1
    r=1
    l=0
    col_indx_c = col_indx[0]
    
    # TODO : loop need to be optimized
    while(val!=0):
        
        if r==1:
            for c in range(col_indx[0]+1,len(diff_col_var)):
                if q3_c_var_val[c]!=0:
                    col_indx_c+=1
                    col_last = col_indx_c
                elif col_indx_c>=len(diff_col_var):
                    col_last = col_indx_c
                    l=1
                    r=0
                    col_indx_c = col_indx[0]
                    break
                else:
                    col_last = col_indx_c
                    l=1
                    r=0
                    col_indx_c = col_indx[0]
                    break
            r=0
            l=1
        if l==1:
            for c in range(0,len(diff_row_var)):#col_indx[0]+1): #,len(diff_row_var)):
                col_indx_c-=1
                if q3_c_var_val[col_indx_c]!=0 and col_indx_c>0:
                    
                    col_first = col_indx_c
                elif col_indx_c<=0:
                    col_first = col_indx_c
                    val=0
                    break
                else:
                    col_first = col_indx_c
                    l=0
                    r=0
                    val=0
                    break
            l=0
            val=0
    
    val = 1
    r=1
    l=0
    
    col_indx_r = row_indx[0]
    while(val!=0):
    
        if r==1:
            for c in range(row_indx[0]+1,len(diff_row_var)):
                if q3_r_var_val[c]!=0:
                    col_indx_r+=1
                    row_last = col_indx_r
                elif col_indx_r>=len(diff_row_var):
                    row_last = col_indx_r
                    l=1
                    r=0
                    col_indx_r = row_indx[0]
                    break
                else:
                    row_last = col_indx_r
                    l=1
                    r=0
                    col_indx_r = row_indx[0]
                    break
            r=0
            l=1
        if l==1:
            for c in range(0,len(diff_row_var)):#row_indx[0]+1): # len(diff_row_var)):
                col_indx_r-=1
                if q3_r_var_val[col_indx_r]!=0 and col_indx_r>0:
                    
                    row_first = col_indx_r
                
                elif col_indx_r<=0:
                    row_first = col_indx_r
                    val=0
                    break

                else:
                    row_first = col_indx_r
                    l=0
                    r=0
                    val=0
                    break
            l=0
            val=0
    try:

        if row_first==row_last:
            row_first   = row_indx[0]
            row_last    = row_indx[0]+5
    except:
        try:
            row_last = row_first
            row_first   = row_indx[0]
            row_last    = row_indx[0]+5
        except:
            
            row_first   = row_last
            row_first   = row_indx[0]
            row_last    = row_indx[0]+5
    try:
        if col_first==col_last:
            col_first   = col_indx[0]
            col_last    = col_indx[0]+5
    except:
        try:
            col_last    = col_first
            col_first   = col_indx[0]
            col_last    = col_indx[0]+5
        except:
            col_first   = col_last
            col_first   = col_indx[0]
            col_last    = col_indx[0]+5
    
    try:
        a          = row_first
        b          = row_last
        c          = col_first
        d          = col_last

        # -- print("row_first, row_last", row_first, row_last)
        # -- print("col_first, col_last", col_first, col_last)
    except:
        col_first, row_first   = col_indx,row_indx
        row_first   = row_indx[0]
        row_last    = row_indx[0]+5
        col_first   = row_indx[0]
        col_last    = row_indx[0]+5

    start_position = (abs(row_first),abs(col_first))
    end_position   = (abs(row_last),abs(col_last))
    #over_est_conts = int(0.1*len(diff_col_var))
    w              = abs(end_position[0] - start_position[0]) #+ over_est_conts
    h              = abs(end_position[1] - start_position[1]) #+ over_est_conts 

    return w,h



def point_of_interest_v1(residual_img, left_rows_i=0.1,left_rows_f=0.9, left_cols_i=0.1,left_cols_f=0.9):

    try:
        rows_img, cols_img,_ = residual_img.shape
    except:
        rows_img, cols_img = residual_img.shape

    #### try mean 
    r,c     = residual_img.shape[0],residual_img.shape[1]
    var_r   = np.var(np.reshape(residual_img,(r,c)),axis=1)
    var_c   = np.var(np.reshape(residual_img,(r,c)),axis=0)
    
    # -- Orignal Variance 
    row_max = np.max(var_r)
    col_max = np.max(var_c)  
    
    
    ### Find peak maxima 
    try:
        var_col_peal_h  = np.max(find_peaks(var_c, height=1)[1]['peak_heights'])
        var_row_peal_h  = np.max(find_peaks(var_r, height=1)[1]['peak_heights'])

        perctage_th     = [0.7,0.65,0.60,0.55,0.50]
        th_var_col      = [var_col_peal_h*i for i in perctage_th]
        th_var_row      = [var_row_peal_h*i for i in perctage_th]

        #for i in th_var_col: print(f"threshod {i} : col_index of BBox = ", np.median(find_peaks(var_c, height=i)[0]))
        #for i in th_var_row: print(f"threshod {i} : row_index of BBox = ", np.median(find_peaks(var_r, height=i)[0]))
        ### End
        ### - - col_loc = value_indx(var_c,col_max)
        ### - - row_loc = value_indx(var_r,row_max)
        col_extrema     = [] 
        row_extrema     = []

        for i in th_var_col: col_extrema.append(np.median(find_peaks(var_c, height=i)[0]))
        for i in th_var_row: row_extrema.append(np.median(find_peaks(var_r, height=i)[0]))

        col_loc         = [int(np.mean(col_extrema))]
        row_loc         = [int(np.mean(row_extrema))]

    except:

        col_max       = np.max(var_c)
        row_max       = np.max(var_r) 
        col_loc       = value_indx(var_c,col_max)
        row_loc       = value_indx(var_r,row_max)

        # -- print(f"BBox Col and Row location:({col_loc} ,{row_loc})")

        #print("Comaprsion between Row and Column Variance Plot")
        #x       = [i for i in range(len(var_r))]
        '''
    
    plt.figure()
    plt.plot(x, var_r, 'r--',label='Row Variance')
    plt.plot(x,var_c,'b--',label='Column Variance')
    plt.legend()
    plt.show()
    '''
    ######
    start_position, end_position = bbox_size_v1(var_c,var_r,col_loc,row_loc,
                                            no_of_inc_i = 30,no_of_inc_f=30)

    return col_loc[0], row_loc[0], start_position, end_position

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (float(boxAArea + boxBArea - interArea)+1)
    # return the intersection over union value
    return iou


def draw_bbox_str_end_v1(orig_img,col_idx=0,row_idx=0,width=100,height=100,
save_fig=False,color=(255,0,0),img_indx =0,path_to_save='./'):

    thickness = 2
    window_name = 'Image'
    try:
        backtorgb = cv2.cvtColor(np.array(orig_img),cv2.COLOR_GRAY2RGB)
    except:
        backtorgb = np.copy(orig_img)
        #pass
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #image = cv2.circle(backtorgb, center_coordinates, radius, color, thickness)
    con_bbox_ad = 0
    start_point = (int(abs(col_idx-width/2)-con_bbox_ad), int(abs(row_idx-height/2)-con_bbox_ad))
    # represents the bottom right corner of rectangle
    end_point   = (int(abs(col_idx+width/2)+con_bbox_ad), int(abs(row_idx+height/2)+con_bbox_ad))
    boxAArea = (end_point[0]-start_point[0] + 1) * (end_point[1]-start_point[1] + 1)
    if boxAArea<200:
        return orig_img
    image = cv2.rectangle(backtorgb, start_point, end_point, color, thickness)

    return image



def point_of_interest_v2(residual_img, left_rows_i=0.1,left_rows_f=0.9, left_cols_i=0.1,left_cols_f=0.9):

    try:
        rows_img, cols_img,_ = residual_img.shape
    except:
        rows_img, cols_img = residual_img.shape
    
    rows_initial = int(left_rows_i*rows_img)
    rows_final   = int(left_rows_f*rows_img)

    cols_initial = int(left_cols_i*cols_img)
    cols_final   = int(left_cols_f*cols_img)

    #residual_img   = residual_img[rows_initial:rows_final,cols_initial:cols_final]
    '''
    var_r   = np.var(residual_img,axis=1)
    var_c   = np.var(residual_img,axis=0)
    '''
    #### try mean 
    r,c     = residual_img.shape[0],residual_img.shape[1]
    var_r   = np.var(np.reshape(residual_img,(r,c)),axis=1)
    var_c   = np.var(np.reshape(residual_img,(r,c)),axis=0)
    
    # -- Orignal Variance 
    row_max = np.max(var_r)
    col_max = np.max(var_c)  
    
    #Omit boundary Variance    
    # row_max = np.max(var_r[rows_initial:rows_final])
    # col_max = np.max(var_c[cols_initial:cols_final])    

    # truncated variance 
    #var_r_t = var_r[rows_initial:rows_final]
    #var_c_t = var_c[rows_initial:rows_final]


    # -- Multpile Boxes
    # col_l_m, row_l_m = find_multiple_boxes_v1(var_c, var_r, th_inc = 0.20,
    #                                            row_i=rows_initial, row_f=rows_final, 
    #                                            col_i=cols_initial, col_f=cols_final )

    # if len(col_l_m)>1 or len(row_l_m)>1:
    #    print("col: ",col_l_m, "row: ",row_l_m)
    # End Multiple Boxes
    ### Find peak maxima 
    try:
        #print(abc)

        if np.max(residual_img)>1:
            h = 1
        else:
            h = 0.0001
        
        var_col_peak_h  = find_peaks(var_c, height=h)[0]
        var_row_peak_h  = find_peaks(var_r, height=h)[0]
        locations       = np.array([])

        for col_loc in var_col_peak_h:
            for row_loc in var_row_peak_h:
                if residual_img[row_loc, col_loc]!=0:
                    #print("row_loc, col_loc ", row_loc, col_loc)
                    if len(locations)>0:
                        locations  = np.vstack((locations,[row_loc, col_loc]))
                    else:
                        locations  = np.array([row_loc, col_loc])
        


        #print(f"BBox Col and Row location:({col_loc} ,{row_loc})")
        
        
        #print("Comaprsion between Row and Column Variance Plot")
        #x       = [i for i in range(len(var_r))]
    except Exception as e:
        #print("Error found :",e)
        mult_bbox = False
        return mult_bbox
    '''
    plt.figure()
    plt.plot(x, var_r, 'r--',label='Row Variance')
    plt.plot(x,var_c,'b--',label='Column Variance')
    plt.legend()
    plt.show()
    '''
    ######
    size_bbox   = np.array([])
    bbox_pre    = True
    
    if locations.ndim==1:
        locations = np.array([locations])

    if locations.size!=0:
        for location in locations:
            
            col_loc, row_loc    = [locations[-1][1]], [locations[-1][0]]
            w, h                = bbox_size_v1(var_c,var_r,col_loc,row_loc) # ,no_of_inc_i = 30,no_of_inc_f=30)
            
            if len(size_bbox)>0:
                size_bbox  = np.vstack((size_bbox,[w, h]))
            else:
                size_bbox  = np.array([w, h])
        
        if size_bbox.ndim==1:
            size_bbox = np.array([size_bbox])
    else:
        bbox_pre   = False

    return locations, size_bbox, bbox_pre #start_position, end_position

def make_bbox_in_range(point):
    point = list(point)
    for i in range(len(point)):
        if point[i]<=0:
            point[i]=2
        if point[i]>=256:
            point[i]=254
    return tuple(point)


def draw_bbox_str_end_v2(orig_img,col_idx=0,row_idx=0,width=100,height=100,
save_fig=False,color=(255,0,0),img_indx =0,path_to_save='./',SSIM_value=0,ssim_th=0):
    
    if SSIM_value<=ssim_th:
        # Blue color in BGR
        #color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        window_name = 'Image'
        try:
            backtorgb = cv2.cvtColor(np.array(orig_img),cv2.COLOR_GRAY2RGB)
        except:
            pass
            backtorgb = orig_img
            #backtorgb = cv2.cvtColor(np.array(orig_img),cv2.COLOR_GRAY2RGB)
        # Draw a rectangle with blue line borders of thickness of 2 px
        # Center coordinates
        center_coordinates = (row_idx,col_idx)
        
        # Radius of circle
        #dist = np.linalg.norm(np.array(start_point)-np.array(end_point))
        #radius = int(dist/2)
        radius  = 30
        
        # Blue color in BGR
        color = (255, 0, 0)
        
        # Line thickness of 2 px
        thickness = 2
        
        # Using cv2.circle() method
        # Draw a circle with blue line borders of thickness of 2 px
        #image = cv2.circle(backtorgb, center_coordinates, radius, color, thickness)
        '''if width<=5 or height<=5:
            con_bbox_ad = 7
        else:
            con_bbox_ad = 0'''
        con_bbox_ad = 10   

        
        start_point = (int(abs(col_idx-width/2)-con_bbox_ad), int(abs(row_idx-height/2)-con_bbox_ad))
        # represents the bottom right corner of rectangle
        end_point   = (int(abs(col_idx+width/2)+con_bbox_ad), int(abs(row_idx+height/2)+con_bbox_ad))

        start_point = make_bbox_in_range(start_point)
        end_point   = make_bbox_in_range(end_point)

        boxAArea = (end_point[0]-start_point[0] + 1) * (end_point[1]-start_point[1] + 1)
        if boxAArea<0: ## 200
            return orig_img
        
        

        image = cv2.rectangle(backtorgb, start_point, end_point, color, thickness)

        # Displaying the image 
        
        #cv2.imshow(window_name, image)
        #plt.figure()
        #plt.imshow(image)
        #if save_fig:
        #    plt.savefig(f'{path_to_save}{img_indx}.png')
        #plt.close()    

        #cv2.waitKey(0) 
        #cv2.destroyAllWindows() 
        return image
    else:
        return orig_img

def median_th(image_res_th, med_img_thres=100):
    median_img = cv2.medianBlur(image_res_th, ksize=5)
    
    # median_img      			= med_filt(image_res_th,size=5) 
    # cv2.imwrite(f'temp_results/{ab_frame_count}.png',residual_color)
    median_img[median_img<(np.max(median_img)*.7)]   = 0  # 0.7 best , .9
    #median_img[median_img<med_img_thres]   = 0 # 100
    return median_img

def thresh_hist_med(out_mask):
    median_img      = cv2.medianBlur(out_mask, ksize=5)
    median_img[median_img<(np.max(median_img)*.95)]   = 0  # 0.7 best , .9
    #median_img[median_img<80]   = 0
    return median_img

def residual_th(res_img, threshold_per = 0.05):
    try:
        rows,cols,_  	= res_img.shape
        residual_abs  	= np.array(np.abs(res_img)*255,dtype=np.uint8).reshape((rows,cols,3))
    except:
        rows,cols     	= res_img.shape
        residual_abs  	= np.array(np.abs(res_img)*255,dtype=np.uint8).reshape((rows,cols,1))
    
    
    unique, counts 	= np.unique(residual_abs, return_counts=True)
    count_th_1      = counts<=(threshold_per*np.max(counts))
    counts_th       = dict(zip(unique, count_th_1))
    image_res_th 	= np.array(residual_abs)
    
    intensity_n_zer = np.array(list(counts_th.values()))*np.array(list(counts_th.keys()))
    intensity_n_zer = intensity_n_zer[np.nonzero(intensity_n_zer)]
    val_th          = np.isin(image_res_th,intensity_n_zer)
    img_th          = residual_abs* val_th
    
    '''
    for i in range(rows):
        for j in range(cols):

            if len(residual_abs.shape)==3:
                value       = residual_abs[i,j,0]
            else:
                value       = residual_abs[i,j]

            counts_int  = counts_th[value]
            if counts_int==0:                             #if gray_img_np[i,j]>=60 and gray_img_np[i,j]<=215:
                image_res_th[i,j]   = 0                           #gray_img_np[i,j] = 255
            else:
                pass
    return image_res_th'''

    return img_th


def noise_removal_module(batch, thresh=0.40):
    #sig = nn.Sigmoid()
    batch_new = batch.detach().cpu().numpy()
    for i in range(batch.shape[0]):
        image_residual_n        =       batch_new[i].reshape(256,256)
        image_residual_th       =       residual_th(image_residual_n,threshold_per=thresh)
        median_img              =       med_filt(image_residual_th,size=3)
        batch_new[i,0,:,:]      =       torch.tensor(median_img[:,:,0]/(np.max(median_img)+0.005),dtype=torch.float32)
    
    return batch_new

def roc_auc_metric(gray_rec_n_r, anomaly_mask):
    gray_rec_n_r_val            =       gray_rec_n_r.detach().cpu().numpy()
    anomaly_mask_val            =       anomaly_mask.detach().cpu().numpy()

    auc_score                   =       np.array([])
    gray_rec_n_r_val_all_img    =       np.array([])
    gray_rec_n_r_val_all_img_b  =       np.array([])

    anomaly_mask_val_all_img    =       np.array([])
    anomaly_mask_val_all_img_b  =       np.array([])

    #dim_img                     =       gray_rec_n_r_val.shape[-1]       

    for i in range(gray_rec_n_r_val.shape[0]):
        gray_rec_n_r_val_img        =       gray_rec_n_r_val[i]
        gray_rec_n_r_val_img        =       gray_rec_n_r_val_img.flatten()
        gray_rec_n_r_val_img_b      =       np.array(gray_rec_n_r_val_img>0, dtype='uint8')
        gray_rec_n_r_val_all_img    =       np.append(gray_rec_n_r_val_all_img,gray_rec_n_r_val_img)
        gray_rec_n_r_val_all_img_b  =       np.append(gray_rec_n_r_val_all_img_b,gray_rec_n_r_val_img_b)

        anomaly_mask_val_img        =       anomaly_mask_val[i]
        anomaly_mask_val_img        =       anomaly_mask_val_img.flatten()
        anomaly_mask_val_img_b      =       np.array(anomaly_mask_val_img>0, dtype='uint8')
        anomaly_mask_val_all_img    =       np.append(anomaly_mask_val_all_img,anomaly_mask_val_img)
        anomaly_mask_val_all_img_b  =       np.append(anomaly_mask_val_all_img_b,anomaly_mask_val_img_b)

        
    try:
        auc_score                       =       roc_auc_score(anomaly_mask_val_all_img, gray_rec_n_r_val_all_img)
        p, r, f, _                      =       precision_recall_fscore_support(anomaly_mask_val_all_img_b, gray_rec_n_r_val_all_img_b,pos_label=None,
                                                 average='weighted')
    except:
        anomaly_mask_val_all_img[-1]    =       1
        print("All Normal Images")
        auc_roc                         =       roc_auc_score(anomaly_mask_val_all_img, gray_rec_n_r_val_all_img)
        p, r, f, _                      =       precision_recall_fscore_support(anomaly_mask_val_all_img_b, gray_rec_n_r_val_all_img_b,pos_label=None,
                                                 average='weighted')
    #auc_score                   =       np.append(auc_score,score)
    #print("Score for each image is : ", auc_score)
    return auc_score, p, r, f


def seg_module(orig_batch, res_batch, th_pix=0.95, th_val=30):
    #sig = nn.Sigmoid()
    batch_new = res_batch.detach().cpu().numpy()
    batch_org = orig_batch.detach().cpu().numpy()
    for i in range(res_batch.shape[0]):
        image_residual_n        =       batch_new[i].reshape(256,256)
        image_residual_th       =       residual_th(image_residual_n,threshold_per=th_pix)
        median_img              =       med_filt(image_residual_th,size=3)
        segments_slic           =       slic(batch_org[i,:,:,:].reshape((256,256,3)), n_segments=250, compactness=8, sigma=1,start_label=1)
        uniq_v                  =       np.unique(segments_slic*np.array(median_img[:,:,0]>th_val,dtype='uint8'))
        seg_img                 =       np.isin(segments_slic,uniq_v)

        croped_img              =       crop_area(batch_org[i,:,:,:],seg_img)
        #batch_new[i,0,:,:]      =       torch.tensor(median_img[:,:,0]/(np.max(median_img)+0.005),dtype=torch.float32)
        if np.max(croped_img)>1:
            #batch_org[i,:,:,:]      =       torch.tensor(croped_img.reshape((3,256,256))/(np.max(croped_img)+0.005),dtype=torch.float32)
            batch_org[i,:,:,:]      =       torch.tensor(croped_img/(np.max(croped_img)+0.005),dtype=torch.float32)
        else:
            batch_org[i,:,:,:]      =       torch.tensor(croped_img,dtype=torch.float32)
    
    return batch_org

def crop_area(orig_img, mask):
    #orig_img        =           orig_img.reshape((256,256,3))
    for i in range(3):
        orig_img[i,:,:]         =        orig_img[i,:,:]*(mask>0)

    return orig_img

def decode_output(output):
    output = output.split()
    for i, v in enumerate(output):
    #print(i)
        output[i] = v.decode()
    return output 

def find_values(results_val, val):
    value_indx = results_val.index(val)+2
    value      = float(results_val[value_indx])
    return value, value_indx


def per_anomaly(perlin_noise, anomaly_img_augmented, image):
    perlin_thr          = np.where(perlin_noise >  np.random.rand(1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            
            #perlin_thr = np.where((anomaly_img_augmented.astype(np.float32)[:,:,0]/255)>0.5, np.ones_like(perlin_noise), np.zeros_like(perlin_noise)) 
        #perlin_thr = np.where(perlin_noise >  np.random.rand(1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
    perlin_thr          = np.expand_dims(perlin_thr, axis=2)

    img_thr             = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

    beta                = torch.rand(1).numpy()[0] * 0.8

    augmented_image     = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (
        perlin_thr)
    
    return augmented_image,perlin_thr
    

def compute_pro_score(amaps: NDArray, masks: NDArray) -> float:

    datas = []
    binary_amaps = np.zeros_like(amaps, dtype=np.bool)

    max_step = 200
    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / max_step

    for th in tqdm(np.arange(min_th, max_th, delta), desc="compute pro"):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                TP_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(TP_pixels / region.area)

        inverse_masks = 1 - masks
        FP_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = FP_pixels / inverse_masks.sum()

        datas.append({"pro": mean(pros), "fpr": fpr, "threshold": th})

    df = pd.DataFrame(datas)
    # df.to_csv("pro_curve.csv", index=False)
    return auc(df["fpr"], df["pro"])

def add_norm_imgs_to_anom_imgs(anomaly_source_paths,image_paths):
    random.shuffle(anomaly_source_paths)
    no_of_anom_imgs             =   len(anomaly_source_paths)
    reduced_imgs                =   int(0.1*no_of_anom_imgs)
    anomaly_source_paths        =   anomaly_source_paths[:reduced_imgs]
    [anomaly_source_paths.append(paths) for paths in image_paths]
    return anomaly_source_paths

def plt_imsave(img):
    plt.imsave('a.png',img)
    return None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def shape_equal(img1,img2):
    img_2_tensor = torch.tensor(img2.cpu().detach().numpy())
    img_2_3d     = torch.tensor(img_2_tensor)
    for i in range(img1.shape[1]-1):
        img_2_3d = torch.cat([img_2_3d, img_2_tensor],dim=1)
    return img_2_3d

    return 

def put_text_img(img, txt='', origin=(0,0), color_txt = (255, 0, 0), thickness_txt=2, 
fontScale = 1, font = cv2.FONT_HERSHEY_SIMPLEX):
    '''customize opencv text function https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    '''
    image = cv2.putText(img, txt, origin, font, 
                   fontScale, color_txt, thickness_txt, cv2.LINE_AA) # Using cv2.putText() method
    return image 

def float_int8_color(img):
    
    if np.max(img)<=1:
        img     =   (abs(img)*255).astype(np.uint8)
    else:
        img     =   abs(img).astype(np.uint8)
    #if (len(img.shape)==2): img   =   cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if  (len(img.shape)==2) or (img.shape[-1]==1): 

        img   =   cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
        #img   =   cv2.cvtColor(img, cv2.cv2.COLOR_BGR2HSV) 

    return img

def pred_result_save(all_imgs, img_indx = 0, show_results=False, gth_available=1,save_res=False,
                    fig_names           =  ''):
    
    combine_img         =   np.array([])

    for indx, img in enumerate(all_imgs):
        all_imgs[indx]  =   float_int8_color(img)
    #out_mask_cv_1       =   np.array(cv2.applyColorMap(np.array(abs(out_mask_cv_1),dtype=np.uint8),cv2.COLORMAP_JET), dtype=np.uint8)
        if      len(combine_img) == 0:  combine_img    =   all_imgs[indx]
        else:   combine_img      =   np.hstack((combine_img, all_imgs[indx]))
    
    combine_img         =   np.array((combine_img), dtype=np.uint8) 

    if show_results:
        cv2.imshow('results', cv2.resize(combine_img, (720,480)))
        cv2.waitKey(10)
    try: os.mkdir('results_th')
    except: pass
    if save_res:
        cv2.imwrite('./results_th/residual_img_{0:0>5}.png'.format(img_indx),combine_img)
    return 
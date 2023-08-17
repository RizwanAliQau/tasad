import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
import cv2 as cv
from scipy import ndimage
from anomly_area import *
import glob 
from utils import *
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
import os


def super_pixel_anomaly(resi_img,orig_img):
    rgba_image = PIL.Image.open(resi_img)
    #rgb_image  = rgba_image.convert('RGB')
    gray_image = rgba_image.convert("L")

    #plt.imshow(gray_image)
    #plt.show()

    gray_img_np = np.array(gray_image)

    ##### Roi
    #col_indx, row_indx,w, h = point_of_interest(gray_img_np)
    #image_with_roi                                  = draw_bbox_str_end(gray_img_np,col_idx=col_indx,
    #                                                    row_idx   = row_indx,
    #                                                   width   = w,
    #                                                    height  = h)
    ######
    # display ROI with graph 
    #plt.figure()
    #plt.imshow(image_with_roi)
    ###

    unique, counts = np.unique(gray_img_np, return_counts=True)
    #plt.figure()
    #plt.hist(unique, unique, weights=counts)
    #plt.show()

    toal_pixels     = np.multiply(gray_img_np.shape[0],gray_img_np.shape[1])
    count_th_1      = counts<=(0.05*toal_pixels)
    #count_th_2      = counts>=(0.0005*toal_pixels)
    #count_th_com    = np.logical_and(count_th_1, count_th_2)
    #counts          = np.multiply(counts, count_th_com)
    #counts_th       = dict(zip(unique, counts))
    counts_th        = dict(zip(unique, count_th_1))

    for i in range(gray_img_np.shape[0]):
        for j in range(gray_img_np.shape[1]):
            value       = gray_img_np[i,j]
            counts_int  = counts_th[value]
            if counts_int==0:                             #if gray_img_np[i,j]>=60 and gray_img_np[i,j]<=215:
                gray_img_np[i,j]   = 0                           #gray_img_np[i,j] = 255
            else:
                pass

                #gray_img_np[i,j] = 0
    #plt.figure()
    #plt.imshow(gray_img_np)


    med_img     = ndimage.median_filter(gray_img_np, size=13) # 13
    #plt.figure()
    #plt.imshow(med_img)
    #plt.figure()
    #plt.imshow(np.multiply(med_img,med_img>=45))
    
    img_th = np.multiply(med_img,med_img>=40) # 45
    img    = img_th>0
    img_bin = np.array(img,np.uint8) # , dtype=np.bin)
    orign_img_patch = gray_image* img
    #plt.figure()
    #plt.imshow(orign_img_patch)
    ### superpixels

    #rgb_image  = rgba_image.convert('RGB')
    rgba_image_o = PIL.Image.open(orig_img)
    gray_image_o = rgba_image_o.convert("L")
    gray_img_o_re = cv.resize(np.array(gray_image_o),(256,256))
    color_img_o_re  = cv.resize(np.array(rgba_image_o),(256,256))
    
    #gradient = sobel(gray_image_o)
    #segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    #segments_quick = quickshift(color_img_o_re, kernel_size=3, max_dist=6, ratio=0.5) # rgba_image_o
    ####
    segments_slic = slic(color_img_o_re, n_segments=250, compactness=8, sigma=1,
                     start_label=1)
    seg_img    = mark_boundaries(color_img_o_re, segments_slic)

    seg_values = []
    img_new_res = np.array(gray_img_o_re)
    for i in range(segments_slic.shape[0]):
        for j in range(segments_slic.shape[1]):
            if img[i,j] ==1:
                seg_values.append(segments_slic[i,j])
            else:
                img_new_res[i,j] = 0


    uniq_v = np.unique(seg_values)

    img_new = np.array(gray_img_o_re) #gray_image_o)
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            if segments_slic[i,j] in uniq_v:
                pass
                #print(i,j)
            else:
                img_new[i,j] = 0
    
    #### superpixel filer 

    count_arr = np.bincount(seg_values)
    corres_val = count_arr[uniq_v]
    orig_val = np.bincount(segments_slic.flatten())
    orig_val_count = orig_val[uniq_v]
    perc = (corres_val/orig_val_count)*100
    perc_th = perc>=np.percentile(perc, 60)
    intens_th = uniq_v*perc_th

    img_new_seg_th = np.array(gray_img_o_re) #gray_image_o)
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            if segments_slic[i,j] in intens_th:
                pass
                    #print(i,j)
            else:
                img_new_seg_th[i,j] = 0

    #images_to_plot  = [color_img_o_re,gray_image_l,gray_image, gray_img_np, med_img, img_th, orign_img_patch, seg_img,img_new_res,img_new,img_new_seg_th,image_with_roi] # , image_bbox] # image_residual
    #    images_titles   = ['Orignal Img','Label Img','Residual Image', 'Threshold Image 1', 'Median Filt Image', 'thresholed Image 2', 'Orignal Image Patch', 'test image with sup_pix','OrigImgCrop', 'OrigImgCrop_Super','OrigImgCrop_Super_th', 'Image_Roi']# , 'Anomaly Localization']
    #    subplot_with_title(images_to_plot,images_titles,save_path=f'./complete_mvtec_data/Results_complete/{class_name}/',img_index=indx,
    #                    savefig=True)
    #print("Number of pixels grater than 150 threshold)",np.sum(np.array(gray_image)>150))
    return img_new_seg_th
    
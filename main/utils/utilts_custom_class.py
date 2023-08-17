import numpy as np
from scipy.signal import find_peaks
import cv2
from utils.utilts_func import *
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

class Anomaly_localization:
    def __init__(self, residual_img):
        self.res_img            = residual_img
        self.count              = 0
        self.bbox_summary       = np.array([])
        self.bbox_cordinates    = np.array([])
        self.boxAArea_sum       = np.array([]) ## v1_add
    def residual_th(self, thresh=0.10):
        rows,cols  	= self.res_img.shape
        residual_abs  	= np.array(np.abs(self.res_img)*255,dtype=np.uint8).reshape((rows,cols,1))
        unique, counts 	= np.unique(residual_abs, return_counts=True)
        count_th_1      = counts<=(thresh*np.max(counts))
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

    def bbox_location(self, col_indx, row_indx, w, h,res_img,frame_count=0):
        start_point = (int(abs(col_indx-w/2)), int(abs(row_indx-h/2)))
        # represents the bottom right corner of rectangle
        end_point   = (int(abs(col_indx+w/2)), int(abs(row_indx+h/2)))
        
        if frame_count>0:
            curr_bbox           = np.array([[col_indx,row_indx,w,h]])
            conts_ad_bbox       = 0
            current_cord        = np.array([[start_point[0]-conts_ad_bbox,start_point[1]-conts_ad_bbox,
                                        end_point[0]+conts_ad_bbox,end_point[1]+conts_ad_bbox]]) 

            self.bbox_summary       = np.vstack((self.bbox_summary,curr_bbox))
            self.bbox_cordinates    = np.vstack((self.bbox_cordinates,current_cord)) 

            iou_area                = bb_intersection_over_union(self.bbox_cordinates[-1],self.bbox_cordinates[-2])

            ### add_v1
            boxAArea                = (end_point[0]-start_point[0] + 1) * (end_point[1]-start_point[1] + 1)
            self.boxAArea_sum       = np.append(self.boxAArea_sum, boxAArea)
            ###
            if iou_area>=0.30:
                col_indx, row_indx,w, h = self.bbox_summary[0]

                
                self.bbox_summary       = np.delete(self.bbox_summary,1,axis=0)
                self.bbox_cordinates    = np.delete(self.bbox_cordinates,1,axis=0)

                

            else:
                boxAArea = (end_point[0]-start_point[0] + 1) * (end_point[1]-start_point[1] + 1)
                if boxAArea<np.mean(self.boxAArea_sum)*0.2 or res_img[row_indx, col_indx]==0: # orignal(200):
                    col_indx, row_indx,w, h = self.bbox_summary[0]
                    self.bbox_summary       = np.delete(self.bbox_summary,1,axis=0)
                    self.bbox_cordinates         = np.delete(self.bbox_cordinates,1,axis=0)
                else:
                    self.bbox_summary       = np.delete(self.bbox_summary,0,axis=0)
                    self.bbox_cordinates    = np.delete(self.bbox_cordinates,0,axis=0)
                
        else:
            self.bbox_summary    = np.array([[col_indx,row_indx,w,h]])     
            self.bbox_cordinates = np.array([[start_point[0],start_point[1],end_point[0],end_point[1]]])   
            ## ad_v1
            boxAArea             = (end_point[0]-start_point[0] + 1) * (end_point[1]-start_point[1] + 1)
            self.boxAArea_sum    = np.append(self.boxAArea_sum, boxAArea)
            ##

        #self.count+=1                 

        return col_indx, row_indx, w, h

    

class bbox_history:
    def __init__(self):
        #self.bbox_cordinates         = np.array([]) uncomment in case of video 
        self.bbox_summary            = np.array([])
        self.conts_ad_bbox           = 10
    
    def bbox_hist(self,locations,bbox_size_mult):
        bbox_cordinates         = np.array([]) ##comment in case of video 
        self.bbox_summary       = np.array([]) ## comment in case of video 
        for i in range(len(locations)):
            loc         = locations[i]
            row_indx    = loc[0]
            col_indx    = loc[1]

            bbox        = bbox_size_mult[i]
            w           = bbox[0]
            h           = bbox[1]    

            start_point = (int(abs(col_indx-w/2)), int(abs(row_indx-h/2)))
            # represents the bottom right corner of rectangle
            end_point   = (int(abs(col_indx+w/2)), int(abs(row_indx+h/2)))

            #current_cord    = np.array([[start_point[0],start_point[1],
            #                                end_point[0],end_point[1]]])
            
            current_cord    = np.array([[start_point[0]-self.conts_ad_bbox,start_point[1]-self.conts_ad_bbox,
                                        end_point[0]+self.conts_ad_bbox,end_point[1]+self.conts_ad_bbox]]) 

            curr_bbox       = np.array([[col_indx,row_indx,w,h]])  

            try:
                self.bbox_cordinates = np.vstack((bbox_cordinates,current_cord)) 
                self.bbox_summary    = np.vstack((self.bbox_summary,curr_bbox))

            except:
                self.bbox_summary    = np.array(curr_bbox) # [[col_indx,row_indx,w,h]])     
                bbox_cordinates      = np.array(current_cord)
            
        return self.bbox_summary, bbox_cordinates
    
    def bb_intersection_over_union(self,boxA, boxB):
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

    
    def remove_duplicate_bbox(self,bbox_cordinates,bbox_summary):
        index_to_delete  = np.array([], dtype='uint8')
        const_bbox       = 0
        check_index  = np.arange(1,len(bbox_cordinates),dtype=int)
        index_complete = np.array([])

        for i in range(len(bbox_cordinates)-1):
            for j in range(i+1,len(bbox_cordinates)): #check_index: # range(i+1,len(bbox_cordinates)):
                
                index_complete  = np.append(index_complete,i)

                iou_area        = self.bb_intersection_over_union(bbox_cordinates[i]+self.const_bbox,bbox_cordinates[j]+self.const_bbox)

                if iou_area>=0.05:
                    index_to_delete = np.append(index_to_delete, i)
                    
        bbox_summary = np.delete(bbox_summary,index_to_delete, axis=0)
        return bbox_summary
    
    def image_bbox_c(self,bbox_summary,out_mask_cv,img_indx=0,obj_name='normal', save_img=False):
        #if not(bbox_pre):
        #    pass
            #bbox_summary  = np.copy(pre_bbox_sum)

        if len(bbox_summary)>1:
            print("Multiple bbox:")

        image_bbox       =  np.copy(out_mask_cv)*255
        
        try:
            bbox_summary = bbox_summary[0:2]
        except:
            pass
        
        for bbox in bbox_summary:


            col_indx, row_indx, w, h    = bbox

            con_bbox_ad = 0

            w 			= w+con_bbox_ad
            h			= h+con_bbox_ad   

            ### bbox draw 
            image_bbox                  = draw_bbox_str_end_v2(image_bbox,col_idx=col_indx, # x_test[img_index]
                                        row_idx   = row_indx,
                                        width   = w,
                                        height  = h)


        #     start_point = (int(abs(col_indx-w/2)-con_bbox_ad), int(abs(row_indx-h/2)-con_bbox_ad))
        #     # represents the bottom right corner of rectangle
        #     end_point   = (int(abs(col_indx+w/2)+con_bbox_ad), int(abs(row_indx+h/2)+con_bbox_ad))
        
        # final_img                                       = np.zeros((out_mask_cv.shape))
        
        # final_img[start_point[1]:end_point[1],start_point[0]:end_point[0]]      = out_mask_cv[start_point[1]:end_point[1],start_point[0]:end_point[0]]

        if save_img:
            plt.imsave(f'./reults_th/{obj_name}_00{img_indx}.png',image_bbox/255)
        
        return image_bbox

import torch
import torch.nn as nn
import torch.nn.functional as F

""" class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss

 """

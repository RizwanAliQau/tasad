import numpy as np
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt

class Anomaly_area():
    def __init__(self,img) -> None:
        self.resi_img = np.array(img)

    
    def hist_threshold(self):

        counts, bins    =   np.histogram(self.resi_img,bins=np.max(self.resi_img))
        values          =   np.round(bins[1:])
        intens_count    =   dict()

        th_count     = np.percentile(counts, 97)
        for i in range(len(values)):
            if counts[i]<=th_count:
                intens_count[int(values[i])] = 1

        return intens_count

    def init_thresh(self,inten_count):
        
        for i in range(self.resi_img.shape[0]):
            for j in range(self.resi_img.shape[1]):
                if self.resi_img[i,j] in inten_count:
                    pass
                else:
                    self.resi_img[i,j] = 0
        
        return self.resi_img

    def med_erosion_oper(self):
        uniq_v_find =   False
        for filt_size in range(3,45,2):

            kernel      = np.ones((filt_size,filt_size),np.uint8)
            erosion     = cv.erode(np.array(self.resi_img),kernel,iterations = 1)
            med_img     = ndimage.median_filter(erosion, size=filt_size)
            # med_img     = ndimage.median_filter(self.resi_img, size=filt_size)
            #kernel      = np.ones((filt_size,filt_size),np.uint8)
            #erosion     = cv.erode(np.array(med_img),kernel,iterations = 1)
            uniq_v      = np.unique(erosion)
            if len(uniq_v)<=30:
                #plt.figure()
                #plt.imshow(erosion)
                #plt.show()
                uniq_v_find = True

                break
        if uniq_v_find:
            pass

        else:
            med_img     = ndimage.median_filter(self.resi_img, size=45)
            kernel      = np.ones((45,45),np.uint8)
            erosion     = cv.erode(np.array(med_img),kernel,iterations = 1)
            uniq_v      = np.unique(erosion)
        
        
        return uniq_v, erosion
    
    def count_occ(self,uniq_v, ero_img):
        count_occ = []
        #uniq_v, ero_img = med_erosion_oper(self)

        for val in uniq_v:
            count_occ.append(sum(sum(val==ero_img)))
            print("val: ",val, " occur: ",count_occ[-1])
        
        return count_occ
    
    def value_indx(self,list_for_index,value_find):

        indices = [i for i, x in enumerate(list_for_index) if x == value_find]
        return indices
    
    def gray_val_th(self,count_occ,uniq_v):
        count_occ_u = np.array(count_occ)
        count_occ.sort() 

        gray_values_th = []
        k= 0 
        if len(count_occ)<20:
            int_range = len(count_occ)
        else:
            int_range = 20
        for i in range(int_range):
            index  = self.value_indx(count_occ_u, count_occ[i])

            if len(index)>1:
                for j in index:
                    if uniq_v[j] in gray_values_th:
                        pass
                    else:
                        if count_occ[i]>=30 and count_occ[i]<=900: 
                            gray_values_th.append(uniq_v[j])
                    
            else:
                if count_occ[i]>=30 and count_occ[i]<=900: 
                    gray_values_th.append(uniq_v[index[0]])
                
            
            if len(gray_values_th)>=10:
                break
        
        return gray_values_th

    def th_final_img(self,eroded_img,gray_values_th):
        for i in range(np.array(eroded_img).shape[0]):
            for j in range(np.array(eroded_img).shape[1]):
                if eroded_img[i,j] in gray_values_th:
                    eroded_img[i,j] = 255
                else:
                    eroded_img[i,j] = 0
        med_img_res     = ndimage.median_filter(eroded_img, size=3)

        return med_img_res
import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils.utilts_custom_class import *


def subplot_with_title(images_to_plot,images_titles, savefig='False',save_path='./',img_index=0, SSIM=''):
    no_of_image     = len(images_to_plot)
    rem             = no_of_image%2 
    if (rem==0):
        
        rows, cols  = int(no_of_image/2), int(no_of_image/2)
    else:
        rows,cols   = int(no_of_image/2)+1, int(no_of_image/2)+1     
   
    fig             = plt.figure(figsize=(40,40))
    #plt.subplot_tool()
    plt.subplots_adjust(hspace=0.5)

    for i in range(len(images_to_plot)):

        
        ax              = fig.add_subplot(rows,cols,i+1) #,constrained_layout=True) #
            
        plt.imshow(images_to_plot[i]) # ,cmap='gray')
        
        
        ax.set_title(images_titles[i],fontsize=20)
    if savefig:
        plt.axis('off')

        if img_index<10: img_index     =  '00' + str(img_index)
        elif img_index<100: img_index  =  '0' +  str(img_index)
        elif img_index>=100: img_index =         str(img_index)

        plt.savefig(save_path+img_index+'.png')

    
    #plt.close()

def value_indx(list_for_index,value_find):
    indices = [i for i, x in enumerate(list_for_index) if x == value_find]
    return indices

def ini_final_indx(indx,no_of_val_inc=30):

    indx_ini     =   indx[0]-no_of_val_inc

    if indx_ini<0:
        
        indx_add = abs(indx_ini)
        indx_ini = 0
    else:
        indx_add = 0
    indx_end     =   indx[0]+no_of_val_inc+indx_add

    return indx_ini, indx_end


def bbox_size(diff_col_var,diff_row_var,col_indx,row_indx,no_of_inc = 30):
    
    # Third quartile (Q3)

    #col_ini, col_end       = ini_final_indx(col_indx,no_of_val_inc= no_of_inc)
    #row_ini, row_end       = ini_final_indx(row_indx,no_of_val_inc= no_of_inc)
    col_ini, col_end        = 0, no_of_inc
    row_ini, row_end        = 0, no_of_inc

    Q3_c = np.percentile(diff_col_var[col_ini:col_end], 60, interpolation = 'midpoint')
        # Third quartile (Q3)
    Q3_r = np.percentile(diff_row_var[row_ini:row_end], 60, interpolation = 'midpoint')

    q3_c_var_val = np.array(diff_col_var>Q3_c,dtype='uint8')
    q3_r_var_val = np.array(diff_row_var>Q3_r,dtype='uint8')

    val = 1
    r=1
    l=0
    col_indx_c = col_indx[0]
    

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
        if row_first==row_last: row_first, row_last   = row_indx[0], row_indx[0]+5
    except:
        row_first, row_last   = row_indx[0], row_indx[0]+5
    try:
        if col_first==col_last: col_first, col_last   = col_indx[0], col_indx[0]+5
    except:
        col_first, col_last   = col_indx[0], col_indx[0]+5

    try:
        a,b =  row_first, row_last
        c,d =  col_first, col_last
    except:
        col_first, row_first   = col_indx,row_indx
        row_first, row_last    = row_indx[0], row_indx[0]+5
        col_first, col_last    = col_indx[0], col_indx[0]+5

    start_position = (abs(row_first),abs(col_first))
    end_position   = (abs(row_last),abs(col_last))
    w              = abs(end_position[0] - start_position[0])
    h              = abs(end_position[1] - start_position[1]) 

    return w,h

def point_of_interest(residual_img):
    var_r   = np.var(residual_img,axis=1)
    var_c   = np.var(residual_img,axis=0)

    row_max = np.max(var_r)
    col_max = np.max(var_c)

    col_loc = value_indx(var_c,col_max)
    row_loc = value_indx(var_r,row_max)
    '''
    print("Comaprsion between Row and Column Variance Plot")
    x       = [i for i in range(len(var_r))]

    plt.plot(x, var_r, 'r--',label='Row Variance')
    plt.plot(x,var_c,'b.',label='Column Variance')
    plt.legend()
    plt.show()
    '''
    ######
    start_position, end_position = bbox_size(var_c,var_r,col_loc,row_loc,no_of_inc = len(var_c))

    return col_loc, row_loc, start_position, end_position

def draw_bbox_str_end(orig_img,col_idx=0,row_idx=0,width=100,height=100,
save_fig=False,color=(255,0,0),img_indx =0,path_to_save='./'):
    
    # Blue color in BGR
    #color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2
    window_name = 'Image'
    try:
        backtorgb = cv2.cvtColor(np.array(orig_img),cv2.COLOR_GRAY2RGB)
    except:
        backtorgb = cv2.cvtColor(np.array(orig_img),cv2.COLOR_GRAY2RGB)
    # Draw a rectangle with blue line borders of thickness of 2 px
    # Center coordinates
    center_coordinates = (row_idx[0],col_idx[0])
    
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
    
    start_point = (int(abs(col_idx[0]-width/2)), int(abs(row_idx[0]-height/2)))
    # represents the bottom right corner of rectangle
    end_point   = (int(abs(col_idx[0]+width/2)), int(abs(row_idx[0]+height/2)))
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
    return start_point, end_point



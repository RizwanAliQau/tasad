import torch
import torch.nn as nn
from seg_model import *
from torch import optim
from loss.loss import FocalLoss, SSIM
from utils import *
import os
from utils.tensorboard_visualizer import TensorboardVisualizer
from torch.utils.data import DataLoader
from data_loaders.data_loader import MVTecTrainDataset
from utils.utilts_custom_class import *
from utils.utilts_func         import *
import cv2
import subprocess
from loss.focal_loss import *
### gloabal variables ----- arg
lr      = 0.0001
#pochs  = 800
###


def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    
    for class_nam in args.class_name:
        cuda_map        =  f'cuda:{args.gpu_id}'
        cuda_id         =  torch.device(cuda_map) 
        base_model_name = 'cas_seg_model_weights_mvtech_'
        wght_file_name  =  base_model_name+class_nam

        visualizer      = TensorboardVisualizer(log_dir=os.path.join(args.log_path, wght_file_name+"/"))    
        cas_model       = Seg_Network(in_channels=3, out_channels=1)
        cas_para        = Seg_Network.get_n_params(cas_model)/1000000 
        print("Number of Parmeters of ReconstructiveSubNetwork", cas_para, "Million")

        cas_model.cuda(cuda_id)
        if args.checkpoint_cas_model=='':
            cas_model.apply(weights_init)
        else:
            cas_model.load_state_dict(torch.load(args.checkpoint_cas_model, map_location=cuda_map))  ##'cuda:0'))

        optimizer       = torch.optim.Adam([{"params": cas_model.parameters(), "lr": args.lr}])
        scheduler       = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)


        loss_l2         = torch.nn.modules.loss.MSELoss()
        loss_ssim       = SSIM(args.gpu_id)
        

        
        dataset         = MVTecTrainDataset(args.data_path+class_nam+'/train' , args.anomaly_source_path, resize_shape=[256, 256])
        dataloader      = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16) # 16


        n_iter          = 0
        prev_pixel_ap   = 0
        prev_pixel_auc  = 0
        prev_image_auc  = 0
        for epoch in range(args.epochs):
            print("Epoch: "+str(epoch))
            for i_batch, sample_batched in enumerate(dataloader):

                training_batch          = sample_batched["image"]
                aug_train_batch         = sample_batched["augmented_image"].cuda(cuda_id)
                anomaly_mask_batch      = sample_batched["anomaly_mask"].cuda(cuda_id)

                output_pred             = cas_model(aug_train_batch)

                l2_loss                 = loss_l2(output_pred,anomaly_mask_batch)
                ssim_loss               = loss_ssim(output_pred, anomaly_mask_batch)

                loss                    = l2_loss + ssim_loss  

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 200 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    #visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    #visualizer.plot_loss(segment_loss, n_iter, loss_name='segment_loss')
                   

                if args.visualize and n_iter % 400 == 0:

                    visualizer.visualize_image_batch(training_batch, n_iter, image_name='batch_input')
                    visualizer.visualize_image_batch(aug_train_batch, n_iter, image_name='batch_augmented')
                    visualizer.visualize_image_batch(anomaly_mask_batch, n_iter, image_name='ground_truth')
                    visualizer.visualize_image_batch(output_pred, n_iter, image_name='out_pred')

                    torch.save(cas_model.state_dict(), os.path.join(args.checkpoint_path, wght_file_name+".pckl"))

                    try:

                        results_val             = subprocess.check_output(f'python3 ./main/test_seg_model.py --gpu_id {args.gpu_id_validation} --model_name  {base_model_name} --data_path {args.data_path} --checkpoint_path {args.checkpoint_path} --both_model 0 --obj_list_all {class_nam}', shell=True)
                        results_val             = decode_output(results_val)
                        curr_pixel_ap,indx      = find_values(results_val, 'AP')
                        curr_pixel_auc,_        = find_values(results_val, 'AUC')
                        curr_image_auc,_        = find_values(results_val[indx:], 'AUC')



                        if ((curr_pixel_auc+curr_pixel_ap+curr_image_auc)/3)>=((prev_pixel_ap+prev_pixel_auc+prev_image_auc)/3):
                            torch.save(cas_model.state_dict(), os.path.join(f"{args.best_model_save_path}", wght_file_name+".pckl"))
                            prev_pixel_ap           = curr_pixel_ap
                            prev_pixel_auc          = curr_pixel_auc
                            prev_image_auc          = curr_image_auc
                        
                        print("Class                        :  ", results_val[7])
                        print("Current pixel AP             :  ", curr_pixel_ap)
                        print("Current pixel AUC            :  ", curr_pixel_auc)
                        print("Current image AUC            :  ", curr_image_auc)

                        print("Saved pix AP value               :  ", prev_pixel_ap)
                        print("Saved pix AUC value              :  ", prev_pixel_auc)
                        print("Saved img UC value               :  ", prev_image_auc)

                    except:
                        print("Model saving not begin")

                n_iter +=1

            scheduler.step()



if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--obj_id', action='store', type=int, required=True)
    parser.add_argument('--bs', action='store', type=int, required=True)
    parser.add_argument('--lr', action='store', type=float, required=True)
    parser.add_argument('--epochs', action='store', type=int, required=True)
    parser.add_argument('--gpu_id', action='store', type=int, default=0, required=False)
    parser.add_argument('--gpu_id_validation', action='store', type=int, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, default='', required=False)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store', type=str, required=True) #action='store_true')
    parser.add_argument('--checkpoint_cas_model', action='store', type=str, required=True)
    parser.add_argument('--class_name', action='store', type=str,nargs='+', required=True)
    parser.add_argument('--best_model_save_path', action='store', type=str, required=True)

    args = parser.parse_args()

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)
import torch
import torch.nn as nn
from seg_model import *
from torch import optim
import os
from utils.tensorboard_visualizer import TensorboardVisualizer
from torch.utils.data import DataLoader
from data_loaders.data_loader import MVTecTrainDataset
from utils.utilts_func         import *
from seg_model import *
import subprocess
from loss.loss import FocalLoss, SSIM
from loss.focal_loss import *
from tqdm import tqdm

def train_on_device(args):

    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)

    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    #CUDA_LAUNCH_BLOCKING=1
    
    for class_name in args.class_name:
        
        args.train_gpu_id   =  ','.join([str(i) for i in args.train_gpu_id])
        cuda_map            =  f'cuda:{args.train_gpu_id}'
        cuda_id             =  torch.device(cuda_map) 
        base_model_name     = 'fas_seg_model_weights_mvtech_'

        visualizer      = TensorboardVisualizer(log_dir=os.path.join(args.log_path, base_model_name+class_name+ "/"))    
        ### cas model
        cas_model       = Seg_Network(in_channels=3, out_channels=1)
        cas_par         = Seg_Network.get_n_params(cas_model)/1000000 
        print("Number of Parmeters of ReconstructiveSubNetwork", cas_par, "Million")
        cas_model.cuda(cuda_id)
        cas_model.load_state_dict(torch.load(f'{args.checkpoint_cas_weights}{class_name}.pckl', map_location=f'{cuda_map}'))
        
        ### fas model
        fas_model       = Seg_Network(in_channels=3, out_channels=1)
        fas_par         = Seg_Network.get_n_params(fas_model)/1000000 
        print("Number of Parmeters of ReconstructiveSubNetwork", fas_par, "Million")
        fas_model.cuda(cuda_id)
        if args.checkpoint_fas_weights=='':
            fas_model.apply(weights_init)
        else:
            fas_model.load_state_dict(torch.load(f'{args.checkpoint_fas_weights}{class_name}.pckl', map_location=f'{cuda_map}'))
   

        optimizer = torch.optim.Adam([{"params": fas_model.parameters(), "lr": args.lr}])
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,[args.epochs*0.8,args.epochs*0.9],gamma=0.2, last_epoch=-1)


        loss_l2 = torch.nn.modules.loss.MSELoss()
        loss_ssim = SSIM(args.train_gpu_id[0])
        #loss_focal = BinaryFocalLoss()

        ### dataset 
        dataset     = MVTecTrainDataset(f'{args.data_path}{class_name}/train/', args.anomaly_source_path, args.anom_choices, resize_shape=[256, 256],include_norm_imgs=1, datatype=f'{args.datatype}')
        dataloader  = DataLoader(dataset, batch_size=args.bs, shuffle=True, num_workers=16) # 16

        n_iter = 0

        best_pix_ap      = 0
        best_pix_auc  = 0
        best_img_auc = 0
        #f_los  = FocalLoss(gamma=2)
        for epoch in tqdm(range(args.epochs)):
            
            for i_batch, sample_batched in enumerate(dataloader):

                input_batch         = sample_batched["image"].cuda(cuda_id)
                aug_batch           = sample_batched["augmented_image"].cuda(cuda_id)
                groudtruth_mask     = sample_batched["anomaly_mask"].cuda(cuda_id)

                cas_output          = cas_model(aug_batch)
                input_batch_fas     = torch.tensor(seg_module(aug_batch,cas_output,th_pix=1, th_val=0)).cuda(cuda_id)
                fas_output          = fas_model(input_batch_fas)

                l2_loss             = loss_l2(fas_output,groudtruth_mask)
                #ssim_loss           = loss_ssim(fas_output, groudtruth_mask)
                #segment_loss        = loss_focal(fas_output, groudtruth_mask) 
                loss                = l2_loss  #+ ssim_loss #+ segment_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.visualize and n_iter % 2 == 0:
                    visualizer.plot_loss(l2_loss, n_iter, loss_name='l2_loss')
                    #visualizer.plot_loss(ssim_loss, n_iter, loss_name='ssim_loss')
                    #visualizer.plot_loss(segment_loss, n_iter, loss_name='focal_loss') 

                if args.visualize and n_iter % 400 == 0:
                    
                    visualizer.visualize_image_batch(input_batch, n_iter, image_name='input_batch')
                    visualizer.visualize_image_batch(aug_batch, n_iter, image_name='cas_input')
                    visualizer.visualize_image_batch(groudtruth_mask, n_iter, image_name='mask_target')
                    visualizer.visualize_image_batch(cas_output, n_iter, image_name='out_cas')
                    visualizer.visualize_image_batch(input_batch_fas, n_iter, image_name='fas_input')
                    visualizer.visualize_image_batch(fas_output, n_iter, image_name='out_fas')
                   
                    torch.save(fas_model.state_dict(), os.path.join('./test_weights/', base_model_name+class_name+ ".pckl"))
                    try:

                        cas_model_name          = 'cas'+base_model_name[3:]
                        results_val             = subprocess.check_output(f'python3 ./main/test_seg_model.py --gpu_id {args.val_gpu_id} --model_name  {cas_model_name} --data_path {args.data_path} --checkpoint_path {args.cas_model_path} --both_model 1 --obj_list_all {class_name}', shell=True)
                        results_val             = decode_output(results_val)
                        curr_pix_ap,_           = find_values(results_val, 'AP')
                        curr_pix_auc,index      = find_values(results_val, 'AUC')
                        curr_img_auc,_          = find_values(results_val[index:], 'AUC')
                        
                        if ((curr_pix_auc+curr_pix_ap+curr_img_auc)/3)>=((best_pix_ap+best_pix_auc+best_img_auc)/3):
                            
                            torch.save(fas_model.state_dict(), os.path.join("./best_weights_model_2/", base_model_name+class_name+".pckl"))
                            best_pix_ap        = curr_pix_ap
                            best_pix_auc       = curr_pix_auc
                            best_img_auc       = curr_img_auc

                            
                        print("Class                    :  ", results_val[7])
                        print("Current AP value         :  ", curr_pix_ap)
                        print("Current Pix AUC value    :  ", curr_pix_auc)
                        print("Current Img AUC value    :  ", curr_img_auc)
                        
                        
                        print("Saved AP value           :  ", best_pix_ap)
                        print("Saved Pix AUC value      :  ", best_pix_auc)
                        print("Saved Img AUC value      :  ", best_img_auc)                 


                    except Exception as e:
                        print("Error found: ", e)
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
    parser.add_argument('--train_gpu_id', action='store', type=str, default=0, required=False)
    parser.add_argument('--val_gpu_id', action='store', type=str, default=0, required=False)
    parser.add_argument('--data_path', action='store', type=str, required=True)
    parser.add_argument('--anomaly_source_path', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_path', action='store', type=str, required=True)
    parser.add_argument('--log_path', action='store', type=str, required=True)
    parser.add_argument('--visualize', action='store', type=str, required=True) #action='store_true')
    parser.add_argument('--checkpoint_cas_weights', action='store', type=str, required=True)
    parser.add_argument('--checkpoint_fas_weights', action='store', type=str, required=True)
    parser.add_argument('--class_name', action='store', type=str,nargs='+', required=True)
    parser.add_argument('--anomaly_type', dest='anom_choices', required=False, help='0 -- for superpixel, 1 -- for perlin and [0,1] for both', nargs='+', type=int, default=[0,1])
    parser.add_argument('--cas_model_path', action='store', type=str, required=True)
    parser.add_argument('--datatype', action='store', type=str, required=True)

    args = parser.parse_args()
    gpu_ids     =   args.train_gpu_id.split(',')
    gpu_id      = [int(i) for i in gpu_ids]
    args.gpu_id = gpu_id     


    

    with torch.cuda.device(args.gpu_id):
        train_on_device(args)
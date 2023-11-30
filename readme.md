# create directories 
    
    ├── best_weights_model_1
    ├── best_weights_model_2
    ├── checkpoints
    ├── data
    ├── logs
    ├── test_weights
    └── weights
# create conda environment
    conda create -n ENVNAME --file requirement.txt
    pip install -r requirement_pip.txt

# for testing 
    - download the weights
        - https://drive.google.com/drive/folders/10Z0MNGY9codk0F-h59roTUr4Xeay2IPO?usp=share_link

# tasad testing 
    python ./main/test_seg_model.py --gpu_id 0 --model_name cas_seg_model_weights_mvtech_ --data_path ./data/ --checkpoint_path ./weights/ --both_model 1 --obj_list_all       carpet,grid,leather,tile,wood,bottle,capsule,pill,transistor,zipper,cable,hazelnut,metal_nut,screw,toothbrush 


# for training 
    - download mvtec dataset
        - https://www.mvtec.com/company/research/datasets/mvtec-ad
    - download the texture dataset put inside anomaly source image 
        - https://www.robots.ox.ac.uk/~vgg/data/dtd/
# cas training 
    python ./main/cas_train.py --gpu_id 0 --gpu_id_validation 0 --obj_id -1 --lr 0.0001 --bs 1 --epochs 4000 --data_path ./data/ --anomaly_source_path                ./anomlay_addition_data/ --checkpoint_path ./test_weights/ --log_path ./logs/ --checkpoint_cas_model "" --visualize True --class_name hazelnut --best_model_save_path ./best_weights_model_1/ 

# fas training 
    python ./main/fas_train.py --train_gpu_id 0 --val_gpu_id 0 --obj_id -1 --lr 0.0001 --bs 1 --epochs 4000 --data_path ./data/ --anomaly_source_path ./anomlay_addition_data/ --cas_model_path ./test_weights/ --checkpoint_path ./checkpoints/ --log_path ./logs/ --checkpoint_cas_weights ./test_weights/cas_seg_model_weights_mvtech_ --checkpoint_fas_weights ./test_weights/fas_seg_model_weights_mvtech_ --visualize True --class_name hazelnut --datatype png 

##### ----- #### 
Thanks DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection ---> https://github.com/VitjanZ/DRAEM for providing their code and model weights  

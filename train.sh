### training TTSR
python main.py --save_dir ./train/HMDB/TTSR \
               --reset True \
               --log_file_name train.log \
               --num_gpu 1 \
               --num_workers 1 \
               --dataset HMDB \
               --image_dataset_dir ./Data_CDVL_LR_MC_uf_2_ps_72_fn_6_tpn_1000.h5 \
               --ref_dataset_dir ./Data_CDVL_HR_uf_2_ps_72_fn_6_tpn_1000.h5 \
               --upsample_factor 2 \
               --n_feats 64 \
               --lr_rate 1e-4 \
               --lr_rate_dis 1e-4 \
               --lr_rate_lte 1e-5 \
               --rec_w 1 \
               --per_w 1e-2 \
               --tpl_w 1e-2 \
               --adv_w 1e-3 \
               --batch_size 9 \
               --num_init_epochs 2 \
               --num_epochs 50 \
               --print_every 1 \
               --save_every 10 \
               --val_every 10 \
               --train_crop_size 18


# ### training TTSR-rec
# python main.py --save_dir ./train/CUFED/TTSR-rec \
#                --reset True \
#                --log_file_name train.log \
#                --num_gpu 1 \
#                --num_workers 9 \
#                --dataset CUFED \
#                --dataset_dir /home/v-fuyang/Data/CUFED/ \
#                --n_feats 64 \
#                --lr_rate 1e-4 \
#                --lr_rate_dis 1e-4 \
#                --lr_rate_lte 1e-5 \
#                --rec_w 1 \
#                --per_w 0 \
#                --tpl_w 0 \
#                --adv_w 0 \
#                --batch_size 9 \
#                --num_init_epochs 0 \
#                --num_epochs 200 \
#                --print_every 600 \
#                --save_every 10 \
#                --val_every 10
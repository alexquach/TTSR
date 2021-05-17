### evaluation
python main.py --save_dir ./eval/HMDB/TTSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset HMDB \
               --image_dataset_dir ./Data_CDVL_LR_MC_uf_2_ps_72_fn_6_tpn_1000.h5 \
               --ref_dataset_dir ./Data_CDVL_HR_uf_2_ps_72_fn_6_tpn_1000.h5 \
               --model_path ./train/HMDB/TTSR/model/model_00010.pt
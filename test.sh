### test
python main.py --save_dir ./test/demo/output/ \
               --dataset HMDB_FLOWNET \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./lr_testing.mp4 \
               --hr_path ./hr_testing.avi \
               --model_path ./TTSR.pt
### test
python main.py --save_dir ./test/demo/output/ \
               --dataset HMDB_FLOWNET \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_path ./mc_testing.mp4 \
               --hr_path ./hr_testing.mp4 \
               --model_path ./TTSR.pt \
               --train-style normal
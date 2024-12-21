conda activate pt

# need change the dataset_folder to the config file
python train.py --config=../config/train/ntu-rsvi.txt --model_config=../config/model/transloc4d.txt --gpu_id 0
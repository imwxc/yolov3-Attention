# command
#--------------------------------train insect--------------------------------
python train.py --model_def config/yolov3-insect.cfg --data_config config/insect.data --batch_size 4 --n_cpu 6

python train.py --model_def config/SEyolov3_4insect.cfg --data_config config/insect.data --batch_size 4 --n_cpu 6

#--------------------------------train Pest24--------------------------------
python train.py --model_def config/yolov3-Pest24.cfg --data_config config/Pest24.data --batch_size 2 --n_cpu 6

#--------------------------------test--------------------------------
python test.py --weights_path checkpoints/yolov3_ckpt_90.pth --model_def config/yolov3-custom.cfg --data_config config/insect.data --class_path data/insect/insect.names


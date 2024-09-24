#CUDA_VISIBLE_DEVICES=1 python train.py -s /home/ddinucci/Desktop/single_view_inpaint/images/jeep_dataset_projections -m output/jeep_4_view_mask -r 1 --eval --iterations 5000 --test_iterations 5000 --port 5053 
#CUDA_VISIBLE_DEVICES=1 python render.py -m ./output/jeep_4_view_mask --skip_train
#CUDA_VISIBLE_DEVICES=1 python metrics.py -m ./output/jeep_4_view_mask

CUDA_VISIBLE_DEVICES=1 python train.py -s /home/ddinucci/Desktop/single_view_inpaint/images/jeep_dataset_projections -m output/tmp --transform_train_path /home/ddinucci/Desktop/single_view_inpaint/images/jeep_dataset_projections/transforms_train.json --transform_test_path /home/ddinucci/Desktop/single_view_inpaint/images/jeep_dataset_projections/transforms_test.json  -r 1 --eval --iterations 5000 --test_iterations 5000 --port 5061
CUDA_VISIBLE_DEVICES=1 python render.py -m ./output/tmp --skip_train
CUDA_VISIBLE_DEVICES=1 python metrics.py -m ./output/tmp


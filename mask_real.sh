path='/home/ddinucci/Desktop/datasets/slerp_tmp'
output_path='./output/duster/ford_focus_8/weights_mask'
#CUDA_VISIBLE_DEVICES=1 python train.py -s ${path} -m ${output_path} --mask_loss --torch_data -r 1 --iterations 10000 --test_iterations 5000 --port 7070 
CUDA_VISIBLE_DEVICES=1 python render.py -m ${output_path} --skip_train --torch_data --eval
CUDA_VISIBLE_DEVICES=1 python metrics.py -m ${output_path}



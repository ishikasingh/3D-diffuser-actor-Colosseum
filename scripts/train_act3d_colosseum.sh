main_dir=Act3d_18Peract_100Demo_multitask

# dataset=data/peract/Peract_packaged/train
# valset=data/peract/Peract_packaged/val
data_dir=/home/ishika/peract_dir/peract/data/train_100
root=/home/ishika/peract_dir/act3d-chained-diffuser
dataset=$root/datasets/packaged/train_100
valset=/projects/katefgroup/datasets/rlbench/diffusion_trajectories_val/

task=basketball_in_hoop,close_box,close_laptop_lid,empty_dishwasher,get_ice_from_fridge,hockey,meat_on_grill,move_hanger,wipe_desk,open_drawer,slide_block_to_target,reach_and_drag,put_money_in_safe,place_wine_at_rack_location,insert_onto_square_peg,stack_cups,turn_oven_on,straighten_rope,setup_chess,scoop_with_spatula


lr=1e-4
num_ghost_points=1000
num_ghost_points_val=10000
B=8
C=120
ngpus=6

CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node $ngpus --master_port $RANDOM \
    main_keypose.py \
    --tasks $task \
    --dataset $dataset \
    --valset $valset \
    --instructions $data_dir \
    --max_episodes_per_task=$max_episodes_per_taskvar \
    --gripper_loc_bounds tasks/74_hiveformer_tasks_location_bounds.json \
    --num_workers 1 \
    --train_iters 600000 \
    --embedding_dim $C \
    --action_dim 8 \
    --use_instruction 1 \
    --weight_tying 1 \
    --gp_emb_tying 1 \
    --val_freq 4000 \
    --exp_log_dir $main_dir \
    --batch_size $B \
    --batch_size_val 1 \
    --cache_size 600 \
    --cache_size_val 0 \
    --variations {0..199} \
    --num_ghost_points $num_ghost_points\
    --num_ghost_points_val $num_ghost_points_val\
    --symmetric_rotation_loss 0 \
    --regress_position_offset 0 \
    --num_sampling_level 3 \
    --lr $lr\
    --position_loss_coeff 1 \
    --cameras left_shoulder right_shoulder wrist front\
    --max_episodes_per_task -1 \
    --run_log_dir act3d_multitask-C$C-B$B-lr$lr
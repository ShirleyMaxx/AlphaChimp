#!/usr/bin/env bash

#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=cma_dno
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --nodes=1

#SBATCH --qos=40gpu
#SBATCH --cpus-per-task=48
#SBATCH --time 72:00:00


#python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" \
#         --nproc_per_node=4 --master_port=22525 tools/inference.py \
#         configs/alphachimp/alphachimp_infer576.py \
#         --vis_mode 'mix' \
#         --gpus 4


#python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
#    --nproc_per_node=8 --master_port=25525 tools/test.py \
#    configs/alphachimp/alphachimp_res256.py \
#    --checkpoint work_dirs/alphachimp/alphachimp_res256.pth

#python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
#    --nproc_per_node=8 --master_port=25525 tools/save_tracking.py \
#    configs/alphachimp/alphachimp_tracking576.py \
#    --checkpoint work_dirs/alphachimp/alphachimp_res576.pth \
#    --gpus 8

python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --nproc_per_node=1 --master_port=25525 \
        tools/evaluate_tracking.py configs/evaluate_tracking.py

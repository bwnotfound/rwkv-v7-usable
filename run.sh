#!/bin/bash
#######################################################################################################################
#
# Run demo-training-prepare.sh with the same MODEL_TYPE & N_LAYER & N_EMBD first
# Or, rename your base model to rwkv-init.pth and put it in the output folder
#
# The trainer will load the last rwkv-*.pth in the folder, such that it can continue from a stopped run
# Therefore check the log (### Loading rwkv-xxx.pth... ###), and make sure you don't have extra rwkv-*.pth there
#
#######################################################################################################################

n_layer=12
hidden_size=768
head_size=64

max_length=512
output_dir="output/RWKV-v7-L"$n_layer"-D"$hidden_size # set output folder
ckpt_path="output/RWKV-v7-L12-D768/pretrained-checkpoint-37000"
tokenizer_path=models/tokenizer
# data_file=data/pretrain_hq.parquet
data_file=data/sft_512_mini.parquet
resume=true # set to true if you want to resume training from the last checkpoint

#######################################################################################################################
#
# Note bsz & lr affects model & training performance
# Small data => use smaller bsz & slightly smaller LR
# Large data => use larger bsz & slightly larger LR
# Larger model => use smaller LR
# Finetuning => use very small LR, such as 1e-5
#
#######################################################################################################################

batch_size=16 # takes ~9G VRAM here => reduce this to save VRAM, increase this for faster speed
accumulate_grad_batches=1
lr_init="6e-4"
lr_final="6e-5"
lr_decay_steps=1000000
gradient_checkpointing=true # 1 => slower, save VRAM; 0 => faster, more VRAM
save_strategy=steps         # save every 10 "miniepochs" (1 miniepoch = 40320 * max_length tokens) => decrease if your GPU is weak
save_steps=1000
save_limits=2
enable_checkpointing=false # TODO: Trainer params. This function is overrided by gradient_checkpointing, please set false
gradient_clip_val=1.0
print_params_info=false 
jit_on=false # please set false for train task because jit mode consume more time in backward. train.py force jit_on = false because jit will affect the grad calc. We should only use jit on inference.

N_NODE=1       # number of nodes
GPU_PER_NODE=1 # number of GPUs per node

DS_BUCKET_MB=4 # set to 2 for consumer GPUs, set to 200 for A100 / H100 (affects speed & vram usage)

python train.py --ckpt_path "$ckpt_path" --tokenizer_path "$tokenizer_path" --wandb "RWKV" --output_dir $output_dir \
    --max_length $max_length --resume $resume --max_epochs 10 \
    --data_file $data_file \
    --num_nodes $N_NODE --train_batch_size_per_device $batch_size --n_layer $n_layer --hidden_size $hidden_size \
    --lr_init $lr_init --lr_final $lr_final --lr_decay_steps $lr_decay_steps --warmup_steps 10 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
    --weight_decay 0.001 --save_strategy $save_strategy --save_steps $save_steps --head_size $head_size \
    --accelerator gpu --devices $GPU_PER_NODE --precision bf16 --strategy deepspeed_stage_2 --gradient_checkpointing $gradient_checkpointing --ds_bucket_mb $DS_BUCKET_MB \
    --accumulate_grad_batches $accumulate_grad_batches --enable_checkpointing $enable_checkpointing --save_limits $save_limits \
    --gradient_clip_val $gradient_clip_val --print_params_info $print_params_info --jit_on $jit_on

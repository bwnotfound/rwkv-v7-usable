########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

from dataclasses import dataclass, field
import os, warnings, datetime
import re
import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, AutoTokenizer, AutoConfig

from lightning import Trainer, seed_everything
import lightning as L
from lightning.pytorch.utilities import rank_zero_info

from src.trainer import train_callback
from src.dataset import CausalLMDataset
from src.utils import load_checkpoint


@dataclass
class ScriptArguments:
    ckpt_path: str = field(default="")
    tokenizer_path: str = field(default="")
    wandb: str = field(default="")  # wandb project name. if "" then don't use wandb
    output_dir: str = field(default="output")
    random_seed: int = field(default=-1)
    data_file: str = field(default="")
    max_epochs: int = field(default=500)
    save_strategy: str = field(default="steps")  # "steps" or "epoch"
    save_steps: int = field(default=1000)
    train_batch_size_per_device: int = field(default=12)

    max_length: int = field(default=1024)
    vocab_size: int = field(default=0)
    n_layer: int = field(default=6)
    hidden_size: int = field(default=512)
    dim_ffn: int = field(default=None)
    head_size: int = field(default=64)  # can try larger values for larger models
    gradient_checkpointing: bool = field(
        default=True
    )  # gradient checkpt: saves VRAM, but slower

    lr_init: float = field(
        default=6e-4
    )  # 6e-4 for L12-D768, 4e-4 for L24-D1024, 3e-4 for L24-D2048
    lr_final: float = field(default=1e-5)
    warmup_steps: int = field(default=-1)  # try 10 if you load a model
    beta1: float = field(default=0.9)
    beta2: float = field(default=0.99)
    adam_eps: float = field(default=1e-18)
    weight_decay: float = field(default=0)  # try 0.1
    lr_decay_steps: int = field(default=10000)

    resume: bool = field(default=False)  # my special pile mode
    ds_bucket_mb: int = field(
        default=200
    )  # deepspeed bucket size in MB. 200 seems enough
    load_partial: bool = field(default=False)
    my_testing: str = field(default="x070")
    save_limits: int = field(default=5)

    accelerator: str = field(default="auto")
    strategy: str = field(default="auto")
    devices: str = field(default="auto")
    num_nodes: int = field(default=1)
    precision: str = field(default="32-true")
    max_epochs: int = field(default=-1)

    enable_checkpointing: bool = field(default=False)
    # num_sanity_val_steps: int = field(default=2)

    accumulate_grad_batches: int = field(default=1)
    gradient_clip_val: float = field(default=1.0)
    gradient_clip_algorithm: str = field(default=None)
    step_offset: int = field(default=0)
    print_params_info: bool = field(default=True)


if __name__ == "__main__":
    rank_zero_info("########## work in progress ##########")

    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    if 0 > 1:  # for highlight
        args = ScriptArguments()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.vocab_size = tokenizer.vocab_size

    if "deepspeed" in args.strategy:
        import deepspeed

    if args.random_seed >= 0:
        print(
            f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n"
            * 3
        )
        seed_everything(args.random_seed)

    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*"
    )
    warnings.filterwarnings(
        "ignore", ".*The progress bar already tracks a metric with the*"
    )

    args.enable_checkpointing = False
    # args.betas = (args.beta1, args.beta2)
    args.real_bsz = (
        int(args.num_nodes) * int(args.devices) * args.train_batch_size_per_device
    )
    os.environ["RWKV_HEAD_SIZE"] = str(args.head_size)

    args.run_name = f"L{args.n_layer} D{args.hidden_size} vocab{args.vocab_size} ctx{args.max_length}"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.resume:  # find latest saved model
        ckpt_list = []
        re_checkpoint = re.compile(r"checkpoint\-(\d+)$")
        for name in os.listdir(args.output_dir):
            if not re.match(re_checkpoint, name):
                continue
            ckpt_list.append(name)
        if len(ckpt_list) == 0:
            args.resume = ""
        else:
            num_list = []
            for name in ckpt_list:
                num_list.append(int(name.split("-")[1]))
            num_list.sort()
            args.resume = os.path.join(args.output_dir, f"checkpoint-{num_list[-1]}")
            args.step_offset = num_list[-1]
    else:
        args.resume = ""

    try:
        deepspeed_version = deepspeed.__version__
    except:
        deepspeed_version = None

    rank_zero_info(str(vars(args)) + "\n")

    if args.lr_final == 0 or args.lr_init == 0:
        rank_zero_info(
            "\n\nNote: lr_final = 0 or lr_init = 0. Using linear LR schedule instead.\n\n"
        )

    assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    if args.precision == "fp32":
        for i in range(10):
            rank_zero_info(
                "\n\nNote: you are using fp32 (very slow). Try bf16 / tf32 for faster training.\n\n"
            )
    if args.precision == "fp16":
        rank_zero_info(
            "\n\nNote: you are using fp16 (might overflow). Try bf16 / tf32 for stable training.\n\n"
        )

    os.environ["RWKV_JIT_ON"] = "1"
    if "deepspeed_stage_3" in args.strategy:
        os.environ["RWKV_JIT_ON"] = "0"  # somehow incompatible

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    if args.precision == "fp32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"

    ########################################################################################################
    from src.model import RWKV, RWKVConfig, OptimConfig

    train_data = CausalLMDataset(args)
    args.vocab_size = train_data.vocab_size

    checkpoint_path = args.resume if args.resume else args.ckpt_path
    if checkpoint_path:
        print(f"########## Load config from {checkpoint_path}... ##########")
        config = RWKVConfig.from_pretrained(checkpoint_path)
    else:
        print(f"########## config from args ##########")
        config = RWKVConfig(
            hidden_size=args.hidden_size,
            n_layer=args.n_layer,
            head_size=args.head_size,
            dim_ffn=args.dim_ffn,
            vocab_size=args.vocab_size,
            max_length=args.max_length,
            gradient_checkpointing=args.gradient_checkpointing,
        )
    optim_config = OptimConfig(
        lr_init=args.lr_init,
        lr_final=args.lr_final,
        warmup_steps=args.warmup_steps,
        beta1=args.beta1,
        beta2=args.beta2,
        adam_eps=args.adam_eps,
        weight_decay=args.weight_decay,
        lr_decay_steps=args.lr_decay_steps,
    )
    model = RWKV(config, optim_config, print_params_info=args.print_params_info)
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        rank_zero_info(f"########## Init weight from scratch ##########")
        model.from_state_dict(model.generate_init_weight(args.accelerator))
    else:
        rank_zero_info(
            f"########## Load state_dict from {args.ckpt_path}... ##########"
        )
        load_checkpoint(checkpoint_path, model, load_partial=args.load_partial)

    trainer = Trainer(
        callbacks=[train_callback(args)],
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        num_nodes=args.num_nodes,
        precision=args.precision,
        enable_checkpointing=args.enable_checkpointing,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
    )

    if trainer.global_rank == 0 and args.print_params_info:
        print(
            "########################## Final model parameters ################################"
        )
        for n in model.state_dict():
            shape = model.state_dict()[n].shape
            s = " ".join([str(_).ljust(5) for _ in shape])
            print(f"{s} {n}")

    if "deepspeed" in args.strategy:
        trainer.strategy.config["zero_optimization"]["allgather_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )
        trainer.strategy.config["zero_optimization"]["reduce_bucket_size"] = (
            args.ds_bucket_mb * 1000 * 1000
        )

    # must set shuffle=False, persistent_workers=False (because worker is in another thread)
    data_loader = DataLoader(
        train_data,
        shuffle=False,
        pin_memory=True,
        batch_size=args.train_batch_size_per_device,
        num_workers=2,
        persistent_workers=False,
        drop_last=True,
        collate_fn=CausalLMDataset.collate_fn,
    )

    rank_zero_info(
        f"""
############################################################################
#
# RWKV-7 {args.precision.upper()} on {args.num_nodes}x{args.devices} {args.accelerator.upper()}, bsz {args.num_nodes}x{args.devices}x{args.train_batch_size_per_device}={args.real_bsz}, {args.strategy} {'with gradient_checkpointing' if args.gradient_checkpointing else ''}
#
# Data = {args.data_file}, ProjDir = {args.output_dir}
#
# Model = {config.n_layer} n_layer, {config.hidden_size} hidden_size, {config.max_length} max_length
#
# Adam = lr {optim_config.lr_init} to {optim_config.lr_final}, warmup {optim_config.warmup_steps} steps, beta ({optim_config.betas}), eps {optim_config.adam_eps}
#
# Found torch {torch.__version__}, recommend latest torch
# Found deepspeed {deepspeed_version}, recommend latest deepspeed
# Found lightning {L.__version__}, recommend latest lightning
#
############################################################################
"""
    )

    trainer.fit(model, data_loader)

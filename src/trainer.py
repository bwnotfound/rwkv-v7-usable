import os, math, time, datetime
from functools import partial
from typing import Mapping
import torch

import lightning as L
from lightning.pytorch.utilities import rank_zero_only

from .utils import save_checkpoint


def my_save(args, trainer, model, save_limits, step=None):
    if step is None:
        step = trainer.global_step + args.step_offset
    fn = None
    if "deepspeed_stage_3" in args.strategy:
        fn = partial(trainer.save_checkpoint, weights_only=True)
    save_checkpoint(
        args.output_dir,
        step,
        model.state_dict(),
        model.config,
        save_limits=save_limits,
        save_fn=fn,
    )


class train_callback(L.Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        args = self.args

        real_step = trainer.global_step + args.step_offset

        # LR schedule
        if args.lr_decay_steps > 0 and real_step > args.warmup_steps:  # cosine decay
            progress = (real_step - args.warmup_steps) / args.lr_decay_steps
            progress = max(0, min(1, progress))
            lr_final_factor = args.lr_final / args.lr_init
            lr_mult = (0.5 + lr_final_factor / 2) + (
                0.5 - lr_final_factor / 2
            ) * math.cos(math.pi * progress)
            lr = args.lr_init * lr_mult
        else:
            lr = args.lr_init
        if real_step < args.warmup_steps:
            lr = lr * (0.01 + 0.99 * real_step / args.warmup_steps)

        for param_group in trainer.optimizers[0].param_groups:
            if param_group["weight_decay"] > 0:
                param_group["weight_decay"] = args.weight_decay
            param_group["lr"] = lr * param_group["my_lr_scale"]

        trainer.my_lr = lr
        trainer.my_wd = args.weight_decay

        if trainer.global_step == 0:
            if trainer.is_global_zero:  # logging
                timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
                trainer.my_loss_sum = 0
                trainer.my_loss_count = 0
                trainer.my_log = open(args.output_dir + "/train_log.txt", "a")
                trainer.my_log.write(f"NEW RUN {timestamp}\n{vars(self.args)}\n")
                try:
                    print(f"\n{trainer.strategy.config}\n")
                    trainer.my_log.write(f"{trainer.strategy.config}\n")
                except:
                    pass
                trainer.my_log.flush()
                if len(args.wandb) > 0:
                    print("wandb init...")
                    import wandb

                    wandb.init(
                        project=args.wandb,
                        name=args.run_name + " " + timestamp,
                        config=args,
                        save_code=False,
                    )
                    trainer.my_wandb = wandb

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        args = self.args
        token_per_step = args.max_length * args.real_bsz
        real_step = trainer.global_step + args.step_offset

        if trainer.is_global_zero:  # logging
            t_now = time.time_ns()
            kt_s = 0
            try:
                t_cost = (t_now - trainer.my_time_ns) / 1e9
                kt_s = token_per_step / t_cost / 1000
                self.log("REAL it/s", 1.0 / t_cost, prog_bar=True, on_step=True)
                self.log("Kt/s", kt_s, prog_bar=True, on_step=True)
            except:
                pass
            trainer.my_time_ns = t_now
            # trainer.my_loss = trainer.my_loss_all.float().mean().item()
            if isinstance(outputs, Mapping):
                trainer.my_loss = outputs["loss"].mean().item()
            else:
                assert isinstance(outputs, torch.Tensor)
                trainer.my_loss = outputs.mean().item()

            trainer.my_loss_sum += trainer.my_loss
            trainer.my_loss_count += 1
            trainer.my_epoch_loss = trainer.my_loss_sum / trainer.my_loss_count
            self.log("lr", trainer.my_lr, prog_bar=True, on_step=True)
            self.log("loss", trainer.my_epoch_loss, prog_bar=True, on_step=True)

            if len(args.wandb) > 0:
                lll = {
                    "loss": trainer.my_loss,
                    "lr": trainer.my_lr,
                    "wd": trainer.my_wd,
                    "Gtokens": real_step * token_per_step / 1e9,
                }
                if kt_s > 0:
                    lll["kt/s"] = kt_s
                trainer.my_wandb.log(lll, step=int(real_step))

        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if args.save_strategy == "steps":
                if real_step % args.save_steps == 0:
                    my_save(
                        args,
                        trainer,
                        pl_module,
                        args.save_limits,
                        step=real_step,
                    )

    def on_train_epoch_start(self, trainer, pl_module):
        args = self.args
        dataset = trainer.train_dataloader.dataset
        if hasattr(dataset, "datasets"):
            dataset = dataset.datasets
        elif hasattr(dataset, "dataset"):
            dataset = dataset.dataset
        # print(f'########## world_size {dataset.world_size} global_rank {dataset.global_rank} real_epoch {dataset.real_epoch} ##########')

    def on_train_epoch_end(self, trainer, pl_module):
        args = self.args
        if (trainer.is_global_zero) or (
            "deepspeed_stage_3" in args.strategy
        ):  # save pth
            if args.save_strategy == "epoch":
                real_step = trainer.global_step + args.step_offset
                my_save(
                    args,
                    trainer,
                    pl_module,
                    args.save_limits,
                    step=real_step,
                )

        if trainer.is_global_zero:  # logging
            trainer.my_log.write(
                f"{trainer.current_epoch} {trainer.my_epoch_loss:.6f} {math.exp(trainer.my_epoch_loss):.4f} {trainer.my_lr:.8f} {datetime.datetime.now()}\n"
            )
            trainer.my_log.flush()

            trainer.my_loss_sum = 0
            trainer.my_loss_count = 0

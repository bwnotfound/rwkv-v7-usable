import os
import re
import shutil
import torch


def get_checkpoint_path_list(root_dir):
    ckpt_list = []
    re_checkpoint = re.compile(r"checkpoint\-(\d+)$")
    for name in os.listdir(root_dir):
        if not re.match(re_checkpoint, name):
            continue
        ckpt_list.append(name)
    if len(ckpt_list) > 0:
        ckpt_list.sort(key=lambda x: int(x.split("-")[1]))
    return ckpt_list


def save_checkpoint(
    output_dir: str,
    step: str,
    state_dict: dict,
    config=None,
    save_limits: int = None,
    save_fn=None,
):
    if not os.path.exists(output_dir):
        raise RuntimeError(
            f"Directory {output_dir} does not exist. Please create it before saving the model."
        )
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
    if save_limits is not None:
        ckpt_path_list = get_checkpoint_path_list(output_dir)
        if len(ckpt_path_list) >= save_limits:
            if int(checkpoint_dir.split("-")[-1]) > int(
                ckpt_path_list[-1].split("-")[-1]
            ):
                shutil.rmtree(os.path.join(output_dir, ckpt_path_list[0]))
            else:
                return
    os.mkdir(checkpoint_dir)
    if save_fn is not None:
        save_fn(os.path.join(checkpoint_dir, "model.pth"))
        return
    torch.save(state_dict, os.path.join(checkpoint_dir, "model.pth"))
    if config is not None:
        config.save_pretrained(checkpoint_dir)


def load_checkpoint(
    checkpoint_dir,
    model,
    load_partial=False,
    dtype=None,
):
    checkpoint_dir = checkpoint_dir.rstrip("/")
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"Directory {checkpoint_dir} does not exist.")
    state_dict = torch.load(
        os.path.join(checkpoint_dir, "model.pth"), map_location="cpu"
    )
    if dtype is not None:
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(dtype)
    model.from_state_dict(state_dict, load_partial=load_partial)

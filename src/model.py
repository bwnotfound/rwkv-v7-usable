########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import logging
from dataclasses import dataclass, field
import torch
import torch.nn as nn
from torch.nn import functional as F

import lightning as L
from lightning.pytorch.strategies import DeepSpeedStrategy

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ.get("RWKV_JIT_ON", "0") == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load


def load_cpp_extension(head_size, chunk_len):
    flags = [
        "-res-usage",
        f"-D_C_={head_size}",
        f"-D_CHUNK_LEN_={chunk_len}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    load(
        name="wind_backstepping",
        sources=[f"cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )


loaded_head_size = None


class WindBackstepping(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, q, k, v, z, b, init_state, meta_tensor):
        B, T, H, C = w.shape
        chunk_len = meta_tensor[0]
        assert (
            T % chunk_len == 0
        ), f"input length({T}) must be divisible by chunk_len({chunk_len})"
        assert all(
            i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b]
        ), f"w: {w.dtype}, q: {q.dtype}, k: {k.dtype}, v: {v.dtype}, z: {z.dtype}, b: {b.dtype}"
        assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
        y = torch.empty_like(v)
        s = torch.empty(
            B, H, T // chunk_len, C, C, dtype=torch.float32, device=w.device
        )  # s == wkv_T
        # init_state shape: (B, H, C, C)
        sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
        torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa, init_state)
        ctx.save_for_backward(w, q, k, v, z, b, s, sa, init_state)
        return y, s[:, :, -1]

    @staticmethod
    def backward(ctx, dy, last_state):
        assert all(i.dtype == torch.bfloat16 for i in [dy]), f"dy: {dy.dtype}"
        assert all(i.is_contiguous() for i in [dy])
        w, q, k, v, z, b, s, sa, init_state = ctx.saved_tensors
        dw, dq, dk, dv, dz, db, dinit_state = [
            torch.empty_like(x) for x in [w, q, k, v, z, b, init_state]
        ]
        torch.ops.wind_backstepping.backward(
            w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db, dinit_state
        )
        return dw, dq, dk, dv, dz, db, dinit_state, None


def RUN_CUDA_RWKV7g(q, w, k, v, a, b, init_state, meta_tensor):
    B, T, HC = q.shape
    chunk_len, head_size = meta_tensor[0], meta_tensor[1]
    assert (
        T % chunk_len == 0
    ), f"input length({T}) must be divisible by chunk_len({chunk_len.item()})"
    q, w, k, v, a, b = [
        i.view(B, T, HC // head_size, head_size) for i in [q, w, k, v, a, b]
    ]
    result = WindBackstepping.apply(w, q, k, v, a, b, init_state, meta_tensor)
    return result[0].view(B, T, HC), result[1]  # result[1] shape: (B, H, C, C)


########################################################################################################
from transformers import PretrainedConfig


class RWKVConfig(PretrainedConfig):

    model_type = "RWKV-7"

    def __init__(
        self,
        hidden_size=512,
        n_layer=6,
        head_size=64,
        dim_ffn=None,
        vocab_size=0,
        max_length=1024,
        chunk_len=16,
        torch_dtype="bf16",
        gradient_checkpointing=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.head_size = head_size
        assert (
            hidden_size % head_size == 0
        ), "hidden_size must be divisible by head_size"
        if dim_ffn is None:
            dim_ffn = int((hidden_size * 3.5) // 32 * 32)
        self.dim_ffn = dim_ffn
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.gradient_checkpointing = gradient_checkpointing
        self.chunk_len = chunk_len
        self.torch_dtype = torch_dtype


@dataclass
class OptimConfig:
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

    def __post_init__(self):
        self.betas = (self.beta1, self.beta2)


class RWKVCache:
    def __init__(self):
        self.cache_states = []
        self.cache_x = []
        self.cache_x_mlp = []

    def get(self, layer_idx):
        result = []
        for item in [self.cache_states, self.cache_x, self.cache_x_mlp]:
            if layer_idx >= len(item):
                result.append(None)
            else:
                result.append(item[layer_idx])

        return result

    def update(self, layer_idx, state=None, x=None, x_mlp=None):
        for item, cache_item in zip(
            [state, x, x_mlp], [self.cache_states, self.cache_x, self.cache_x_mlp]
        ):
            if item is None:
                continue
            if layer_idx >= len(cache_item):
                cache_item.append(item)
            else:
                cache_item[layer_idx] = item


class RWKV_Tmix_x070(MyModule):
    def __init__(self, config: RWKVConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.head_size = config.head_size
        self.n_head = config.hidden_size // self.head_size
        assert config.hidden_size % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = config.hidden_size

        with torch.no_grad():
            ratio_0_to_1 = layer_idx / (config.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.9 * ratio_1_to_almost0) + 0.4 * ratio_0_to_1)
            )
            self.x_v = nn.Parameter(
                1.0 - (torch.pow(ddd, 0.4 * ratio_1_to_almost0) + 0.6 * ratio_0_to_1)
            )
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            D_DECAY_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            decay_speed = torch.ones(C)
            for n in range(C):
                decay_speed[n] = -7 + 5 * (n / (C - 1)) ** (
                    0.85 + 1.0 * ratio_0_to_1**0.5
                )
            self.w0 = nn.Parameter(
                decay_speed.reshape(1, 1, C) + 0.5
            )  # !!! 0.5 comes from F.softplus !!!

            D_AAA_LORA = max(32, int(round((1.8 * (C**0.5)) / 32) * 32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(torch.zeros(1, 1, C))

            D_MV_LORA = max(32, int(round((1.3 * (C**0.5)) / 32) * 32))  # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 1.0)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round((0.6 * (C**0.8)) / 32) * 32))  # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.ones(1, 1, C) * 0.85)
            self.k_a = nn.Parameter(torch.ones(1, 1, C))
            self.r_k = nn.Parameter(torch.zeros(H, N))

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward_script_part1(self, x, v_first, last_x):
        B, T, C = x.size()
        H = self.n_head
        if last_x is None:
            xx = self.time_shift(x) - x
        else:
            _x = torch.cat([last_x, x], dim=1)
            xx = self.time_shift(_x) - _x
            xx = xx[:, 1:, :]

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        )  # soft-clamp to (-inf, -0.5)
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_idx == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        return r, w, k, v, -kk, kk * a, g, v_first

    @MyFunction
    def forward_script_part2(self, x, r, k, v, g):
        B, T, C = x.size()
        H = self.n_head
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x

    def rwkv_forward_pytorch(self, w, q, k, v, a, b, chunk_len):
        """
        纯 PyTorch 实现的 forward 操作，其计算流程与 CUDA 版本一致。

        输入张量的形状均为 (B, T, H, C)，其中：
        w, q, k, v, a, b 均为 torch.bfloat16 类型
        输出：
        y    : (B, T, H, C) ，同样以 bfloat16 存储，实际上每个时间步输出是一个标量（复制到每个通道）
        s_out: (B, H, T//chunk_len, C, C) ，在每个 chunk 的末尾保存状态快照（每行都相同）
        sa_out: (B, T, H, C) ，每个时间步保存的 sa（标量复制到所有通道）
        """
        B, T, H, C = w.shape
        device = w.device

        # 初始化输出张量
        y = torch.empty((B, T, H, C), dtype=torch.bfloat16, device=device)
        sa_out = torch.empty((B, T, H, C), dtype=torch.float32, device=device)

        state = torch.zeros(B, H, C, dtype=torch.float32, device=device)
        for t_idx in range(T):
            # 将当前时间步的各个向量转换为 float32，并做必要的变换：
            # 对 w 进行: exp(-exp(w))
            q_vec = q[:, t_idx].float()  # shape: (C,)
            w_vec = torch.exp(-torch.exp(w[:, t_idx].float()))
            k_vec = k[:, t_idx].float()
            a_vec = a[:, t_idx].float()
            b_vec = b[:, t_idx].float()

            # 计算 sa = a · state（标量）
            sa_scalar = (a_vec * state).sum()
            # 将 sa 标量复制到长度为 C 的向量存储到 sa_out 中
            sa_out[:, t_idx].fill_(sa_scalar.item())

            # 取出 v 对应的向量（float32）
            v_vec = v[:, t_idx].float()

            # 更新状态向量：逐元素更新
            state = state * w_vec + sa_scalar * b_vec + k_vec * v_vec

            # 计算 y = q · state（标量），并复制到输出向量的每个元素
            y_scalar = (q_vec * state).sum()
            y[:, t_idx].fill_(y_scalar.item())

        return y, sa_out

    def step_forward(self, x, v_first, meta_tensor, cache: RWKVCache = None):
        pass

    def forward(self, x, v_first, meta_tensor, cache: RWKVCache = None):
        B, T, C = x.size()
        H = self.n_head
        cache_state, last_x, _ = (
            cache.get(self.layer_idx) if cache else (None, None, None)
        )
        if cache_state is None:
            init_state = torch.zeros(
                B, H, C, C, dtype=torch.float32, device=x.device
            )  # 视作wkv_T
        else:
            init_state = cache_state

        cache_x = x
        r, w, k, v, _kk, _kka, g, v_first = self.forward_script_part1(
            x, v_first, last_x
        )
        x, last_state = RUN_CUDA_RWKV7g(r, w, k, v, _kk, _kka, init_state, meta_tensor)
        if cache is not None:
            cache.update(self.layer_idx, last_state, cache_x[:, -1:, :])
        x = self.forward_script_part2(x, r, k, v, g)
        return x, v_first


########################################################################################################


class RWKV_CMix_x070(MyModule):
    def __init__(self, config: RWKVConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_idx / config.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, config.hidden_size)
            for i in range(config.hidden_size):
                ddd[0, 0, i] = i / config.hidden_size
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(config.hidden_size, config.hidden_size * 4, bias=False)
        self.value = nn.Linear(config.hidden_size * 4, config.hidden_size, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (config.hidden_size**0.5), 0.5 / (config.hidden_size**0.5)
        )
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x, cache: RWKVCache = None):
        if cache is None:
            x_mlp = None
        else:
            _, _, x_mlp = cache.get(self.layer_idx)
        if x_mlp is None:
            xx = self.time_shift(x) - x
        else:
            _x = torch.cat([x_mlp, x], dim=1)
            xx = self.time_shift(_x) - _x
            xx = xx[:, 1:, :]
        if cache is not None:
            cache.update(self.layer_idx, None, None, x[:, -1:, :])

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, config: RWKVConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)

        if self.layer_idx == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size)

        self.att = RWKV_Tmix_x070(config, layer_idx)
        self.ffn = RWKV_CMix_x070(config, layer_idx)

    def forward(self, x, v_first, meta_tensor, cache: RWKVCache = None):
        if self.layer_idx == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first, meta_tensor, cache=cache)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x), cache=cache)
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(L.LightningModule):
    def __init__(
        self,
        config: RWKVConfig,
        optim_config: OptimConfig = None,
        print_params_info=True,
    ):
        super().__init__()
        self.config = config
        self.optim_config = optim_config
        assert config.hidden_size % 32 == 0
        assert config.dim_ffn % 32 == 0

        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)

        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])

        self.ln_out = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.print_params_info = print_params_info

        self.meta_tensor = torch.tensor(
            [self.config.chunk_len, self.config.head_size],
            dtype=torch.long,
            device="cpu",
        )

        global loaded_head_size
        if loaded_head_size != config.head_size:
            if loaded_head_size is not None:
                logging.warning(
                    f"head_size changed from {loaded_head_size} to {config.head_size}, reloading cpp extension."
                )
            load_cpp_extension(config.head_size, config.chunk_len)
            loaded_head_size = config.head_size

    def configure_optimizers(self):
        config = self.optim_config

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (config.weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.print_params_info:
            if self.trainer.is_global_zero:
                print("decay", lr_decay, "\n")
                print("1x", lr_1x, "\n")
                print("2x", lr_2x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
        ]

        if config.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": config.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=config.lr_init,
                    betas=config.betas,
                    eps=config.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=config.lr_init,
                betas=config.betas,
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=config.lr_init,
                    betas=config.betas,
                    eps=config.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=config.lr_init,
                betas=config.betas,
                eps=config.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def forward(self, idx, cache: RWKVCache = None):
        config = self.config
        B, T = idx.size()
        assert (
            T <= config.max_length
        ), f"Cannot forward, get seq_len: {T}, exceed max_length: {config.max_length}."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if config.gradient_checkpointing:
                x, v_first = deepspeed.checkpointing.checkpoint(
                    block, x, v_first, cache=cache
                )
            else:
                x, v_first = block(x, v_first, self.meta_tensor, cache=cache)

        x = self.ln_out(x)
        x = self.head(x)
        return x

    def generate(self, idx, max_length):
        pass

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        logits = self(idx)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return L2Wrap.apply(loss, logits)

    # def training_step_end(self, batch_parts):
    #     all = self.all_gather(batch_parts)
    #     if self.trainer.is_global_zero:
    #         self.trainer.my_loss_all = all

    def generate_init_weight(self, accelerator):
        log = self.print_params_info
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s = " ".join([str(_).ljust(5) for _ in shape])
            if log:
                print(f"{s} {n}", end="")

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (1 + int(n.split(".")[1])) / self.config.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                if log:
                    print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                if log:
                    print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.config.vocab_size > self.config.hidden_size:
                    scale = 0.5 * math.sqrt(
                        self.config.vocab_size / self.config.hidden_size
                    )
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                if log:
                    print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1
                if log:
                    print(f" [scale {scale}]")

                if accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if self.config.torch_dtype == "fp16":
                m[n] = m[n].half()
            elif self.config.torch_dtype == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()
        if log:
            print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m

    def from_state_dict(self, state_dict, load_partial=False):
        load_keys = list(state_dict.keys())
        for k in load_keys:
            if k.startswith("_forward_module."):
                state_dict[k.replace("_forward_module.", "")] = state_dict[k]
                del state_dict[k]
        if load_partial:
            load_keys = state_dict.keys()
            for k in self.state_dict():
                if k not in load_keys:
                    state_dict[k] = self.state_dict()[k]
        self.load_state_dict(state_dict)

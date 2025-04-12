########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info


class CausalLMDataset(Dataset):
    def __init__(self, args):
        import pandas as pd

        self.args = args

        self.vocab_size = args.vocab_size
        rank_zero_info(
            f"Current vocab size = {self.vocab_size} (make sure it's correct)"
        )

        data_file = args.data_file
        if os.path.isdir(data_file):
            data_file = [
                os.path.join(data_file, f)
                for f in os.listdir(data_file)
                if f.endswith(".parquet")
            ]
        elif os.path.isfile(data_file):
            data_file = [data_file]
        else:
            raise RuntimeError(f"Invalid data_file: {data_file}")

        self.df = pd.concat([pd.read_parquet(f) for f in data_file], ignore_index=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_ids, labels = row["input_ids"], row["labels"]
        assert isinstance(input_ids, np.ndarray)
        assert isinstance(labels, np.ndarray)
        assert len(input_ids.shape) == 1
        assert input_ids.shape == labels.shape
        assert len(input_ids) <= self.args.max_length

        x = torch.tensor(input_ids, dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)

        return x, y

    @staticmethod
    def collate_fn(batch):
        x = torch.nn.utils.rnn.pad_sequence(
            [item[0] for item in batch], batch_first=True, padding_value=0
        )
        y = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in batch], batch_first=True, padding_value=-100
        )
        CHUNK_LEN = 16
        if x.shape[1] % CHUNK_LEN != 0:
            # pad x, y to multiple of CHUNK_LEN
            pad_len = CHUNK_LEN - (x.shape[1] % CHUNK_LEN)
            x = torch.nn.functional.pad(x, (0, pad_len), value=0)
            y = torch.nn.functional.pad(y, (0, pad_len), value=-100)
        return x, y

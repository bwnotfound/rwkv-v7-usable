import json
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
import shutil
import gc
import numpy as np
from tqdm import tqdm
import pandas as pd

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer


def encode_batch(tokenizer, batch, max_length):
    if "text" in batch[0]:
        texts = [x["text"] for x in batch]
        encoded = tokenizer(
            texts, max_length=max_length, truncation=True, add_special_tokens=False
        )["input_ids"]
        data = [np.array(x) for x in encoded]
        result = [{"input_ids": out[:-1], "labels": out[1:]} for out in data]
    elif "conversations" in batch[0]:
        conversations_list = [x["conversations"] for x in batch]
        prompt_list, response_list = [], []
        for conversations in conversations_list:
            prompt = []
            assert conversations[-1]["role"] == "assistant"
            for conversation in conversations[:-1]:
                role, content = conversation["role"], conversation["content"]
                prompt.append(f"{role}: {content}")
            prompt = "\n".join(prompt) + "\nassistant: "
            response = f"{conversations[-1]['content']}"
            prompt_list.append(prompt)
            response_list.append(response)
        prompt_ids = tokenizer(
            prompt_list,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        response_ids = tokenizer(
            response_list,
            max_length=max_length,
            truncation=True,
            add_special_tokens=False,
        )["input_ids"]
        input_ids, labels = [], []
        for prompt, response in zip(prompt_ids, response_ids):
            if len(response) >= max_length:
                continue
            if len(prompt) + len(response) > max_length:
                prompt = prompt[-(max_length - len(response)) :]
            input_ids.append(prompt + response)
            labels.append(
                [-100] * (len(prompt) - 1) + response + [tokenizer.eos_token_id]
            )
        input_ids = [np.array(x) for x in input_ids]
        labels = [np.array(x) for x in labels]
        result = [
            {"input_ids": out, "labels": label} for out, label in zip(input_ids, labels)
        ]
    else:
        raise ValueError("Invalid data format")

    return result


def run(file_path, batch_size, tokenizer_path, max_length, queue):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    wait_list = []
    cache = []
    for line in lines:
        cache.append(line)
        if len(cache) >= batch_size:
            cache = [json.loads(x) for x in cache]
            wait_list.append(cache)
            cache = []
    if len(cache) > 0:
        cache = [json.loads(x) for x in cache]
        wait_list.append(cache)

    result = []
    for batch in wait_list:
        result.extend(encode_batch(tokenizer, batch, max_length))
        queue.put(len(batch))
    return result


class TqdmThread(Thread):
    def __init__(self, total, queue):
        super().__init__()
        self.queue = queue
        self.total = total
        self.tqdm = tqdm(total=total, desc="Processing")

    def run(self):
        cnt = 0
        while cnt < self.total:
            result = self.queue.get()
            self.tqdm.update(result)
            cnt += result


"""
data format:
{"text": "Hello world!"}
{"text": "This is a demo."}
...
"""

if __name__ == "__main__":
    tokenizer_path = "models/tokenizer"
    # input_path = "data/pretrain_hq.jsonl"
    input_path = "data/pretrain_hq.jsonl"
    output_dir = "data"
    tem_dir = os.path.join(output_dir, "_TEMP")
    os.makedirs(tem_dir, exist_ok=True)
    max_length = 512
    batch_size = 4000
    pool_size = 8

    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".parquet"
    )

    tqdm.write(f"### Convert {input_path} to {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines = [x for x in lines if len(x) > 0]

    block_size = len(lines) // pool_size
    for i in tqdm(range(pool_size), desc="Splitting data to temporary disk"):
        start = i * block_size
        end = (i + 1) * block_size if i != pool_size - 1 else len(lines)
        with open(os.path.join(tem_dir, f"{i}.txt"), "w", encoding="utf-8") as f:
            f.writelines(lines[start:end])

    tqdm.write(f"### Splitting done. Data length: {len(lines)}")
    queue = multiprocessing.Manager().Queue()
    TqdmThread(len(lines), queue).start()
    del lines
    gc.collect()
    tqdm.write(f"### Start converting...")

    df_rows = []
    futures = []
    with ThreadPoolExecutor(pool_size) as pool:
        for i in range(pool_size):
            file_path = os.path.join(tem_dir, f"{i}.txt")
            future = pool.submit(
                run,
                file_path,
                batch_size,
                tokenizer_path,
                max_length,
                queue,
            )
            futures.append(future)
        for future in futures:
            df_rows.extend(future.result())
    df = pd.DataFrame(df_rows)
    df.to_parquet(output_path, index=False)

    tqdm.write(f"### Convert done. Data length: {len(df)}")
    shutil.rmtree(tem_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tqdm.write("### Verifying result...")
    verify_df = df.sample(n=3)
    for index, row in verify_df.iterrows():
        input_ids = row["input_ids"]
        labels = row["labels"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        tqdm.write(f"Text: {text}")
        tqdm.write(f"Input IDs: {input_ids}")
        tqdm.write(f"Labels: {labels}")
        tqdm.write("-" * 40)

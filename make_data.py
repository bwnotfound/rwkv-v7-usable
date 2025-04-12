import json
import os
import multiprocessing
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


"""
data format:
{"text": "Hello world!"}
{"text": "This is a demo."}
...
"""

if __name__ == "__main__":
    tokenizer_path = "models/tokenizer"
    # input_path = "data/pretrain_hq.jsonl"
    input_path = "/root/autodl-tmp/sft_512_mini.jsonl"
    output_dir = "data"
    max_length = 512
    batch_size = 4000

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    output_path = os.path.join(
        output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".parquet"
    )

    print(f"### Convert {input_path} to {output_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    cache = []
    df_rows = []
    wait_list = []
    for line in tqdm(lines):
        cache.append(line)
        if len(cache) >= batch_size:
            cache = [json.loads(x) for x in cache if len(x) > 0]
            wait_list.append(cache)
            cache = []
            # outs = tokenizer(cache, max_length=max_length, truncation=True)["input_ids"]
            # outs = [np.array(x) for x in outs]
            # for out in outs:
            #     df_rows.append({"input_ids": out[:-1], "labels": out[1:]})
            # cache = []
    futures = []
    with multiprocessing.Pool() as pool:
        for batch in wait_list:
            futures.append(
                pool.apply_async(
                    encode_batch,
                    (tokenizer, batch, max_length),
                )
            )
        for future in tqdm(futures):
            df_rows.extend(future.get())
    df = pd.DataFrame(df_rows)
    df.to_parquet(output_path, index=False)

    print("### Convert done. Data length:", len(df))

    print("### Verifying result...")
    verify_df = df.sample(n=3)
    for index, row in verify_df.iterrows():
        input_ids = row["input_ids"]
        labels = row["labels"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"Text: {text}")
        print(f"Input IDs: {input_ids}")
        print(f"Labels: {labels}")
        print("-" * 40)

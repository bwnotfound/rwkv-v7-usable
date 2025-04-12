### This repository is extended from https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7/train_temp, with the following key enhancements:

1. upgrade to the latest PyTorch Lightning, means no `pytorch-lightning` but `lightning`
2. use huggingface tokenizer
3. change the data process so you can pass `input_ids` and `labels` for sft compatiblity
4. add SFT (Supervised Fine-Tuning) capabilities
5. inference available

### Run Code
1. ```shell
    # it is recommanded to install mannually, just install the latest lib.
    pip install -r requirements.txt
    ```
2. modify the parameters of make_data.py and run `python make_data.py` to generate data.
3. run `./run.sh` to train.
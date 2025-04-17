import os

# os.environ["RWKV_JIT_ON"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from threading import Event

import torch
import gradio as gr
from transformers import AutoTokenizer

from src.model import RWKV, RWKVConfig, RWKVCache
from src.utils import load_checkpoint

stop_event = Event()


def generate(
    prompt,
    max_new_tokens,
    temperature,
    top_k,
    top_p,
):
    device = model.device
    stop_event.clear()
    input_ids = tokenizer([prompt], return_tensors="pt").to(device)["input_ids"]
    assert input_ids.shape[0] == 1
    generate_length = 0
    cache = RWKVCache()
    print(input_ids.shape)
    while not stop_event.is_set():
        seq_len = input_ids.shape[1]
        model_input_ids = input_ids[:, cache.get_seq_length() : seq_len]
        print(
            f"input_ids len: {seq_len}, model_input_ids len: {model_input_ids.shape[1]}"
        )
        assert model_input_ids.shape[1] != 0, f"input_ids length: {seq_len}"
        output = model(model_input_ids, cache=cache)
        output = output[:, -1].unsqueeze(1)
        output = torch.nn.functional.softmax(output / temperature, dim=-1).argmax(
            dim=-1
        )
        assert output.numel() == 1
        output_value = output.item()
        if output_value == tokenizer.eos_token_id:
            break
        output_str = tokenizer.decode(
            output_value, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        yield output_str
        input_ids = torch.cat([input_ids, output], dim=1)
        assert (
            input_ids.shape[1] == seq_len + 1
        ), f"input_ids length: {input_ids.shape[1]}"
        generate_length += 1
        if generate_length >= max_new_tokens:
            break


def chat_stream(
    message: str,
    history: list[list[str]],
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    messages = []
    messages.extend(history)

    messages.append({"role": "user", "content": message})

    prompt = [f'{item["role"]}: {item["content"]}\n' for item in messages]
    prompt = "".join(prompt)
    prompt += "assistant: "

    full_response = ""

    for text in generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    ):
        if text:
            full_response += text
            # print(text, end="", flush=True)
            yield full_response
    print(f"\nfinish the response.")


def user(user_message, history):
    if user_message == "":
        return "", history
    return "", history + [{"role": "user", "content": user_message}]


def bot(
    history,
    system_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
):
    if not history:
        return history

    user_message = history[-1]["content"]

    history.append({"role": "assistant"})
    for response in chat_stream(
        user_message,
        history[:-1],
        system_prompt,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
    ):
        history[-1]["content"] = response
        yield history


def clear_history():
    return []


def interrupt():
    stop_event.set()


def main():
    default_device = "cpu"
    with gr.Blocks(css="footer {visibility: hidden}") as app:
        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(type="messages", height=500, label="对话记录")

                with gr.Row():
                    message = gr.Textbox(
                        show_label=False,
                        placeholder="Send message here...",
                        container=False,
                        scale=9,
                    )
                    send_btn = gr.Button("发送", scale=1)
                    interrupt_btn = gr.Button("中断", scale=1)

            with gr.Column(scale=1):
                system_prompt = gr.Textbox(
                    label="system prompt",
                    placeholder="Set character of AI...",
                    value="",
                )
                max_new_tokens = gr.Slider(
                    minimum=1,
                    maximum=32768,
                    value=256,
                    step=1,
                    label="Max New Tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    label="Temperature",
                )
                top_k = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Top K",
                )
                top_p = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.95,
                    step=0.01,
                    label="Top P",
                )
                device_choice = gr.Dropdown(
                    label="Device",
                    choices=["cpu", "cuda"],
                    value=default_device,
                    info="Device to run the model on.",
                )

                clear = gr.Button("Clear history")

        def register_chat_component(fn):
            fn(user, [message, chatbot], [message, chatbot], queue=False).then(
                bot,
                [
                    chatbot,
                    system_prompt,
                    max_new_tokens,
                    temperature,
                    top_k,
                    top_p,
                ],
                chatbot,
            )

        register_chat_component(message.submit)
        register_chat_component(send_btn.click)
        clear.click(clear_history, None, chatbot)
        interrupt_btn.click(interrupt)
    app.queue(default_concurrency_limit=4)
    return app


if __name__ == "__main__":
    ckpt_path = "output/RWKV-v7-L12-D768/sft-checkpoint-75000"
    tokenizer_path = "models/tokenizer"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    config = RWKVConfig.from_pretrained(ckpt_path)
    assert isinstance(config, RWKVConfig)
    config.gradient_checkpointing = False
    config.max_length = 1e10
    model = RWKV(config, print_params_info=False)
    # model.from_state_dict(model.generate_init_weight("gpu"))
    model.to(torch.bfloat16)
    load_checkpoint(ckpt_path, model, dtype=torch.bfloat16)
    model.cuda()
    model.eval()

    app = main()
    app.launch()

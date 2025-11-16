from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import torch


def get_base_model():
    # 4bit pre quantized models we support for 4x faster downloading + no OOMs.
    fourbit_models = [
        "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",  # Llama 3.2 vision support
        "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
        "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit",  # Can fit in a 80GB card!
        "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

        "unsloth/Pixtral-12B-2409-bnb-4bit",  # Pixtral fits in 16GB!
        "unsloth/Pixtral-12B-Base-2409-bnb-4bit",  # Pixtral base model

        "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",  # Qwen2 VL support
        "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
        "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

        "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",  # Any Llava variant works!
        "unsloth/llava-1.5-7b-hf-bnb-4bit",
    ]  # More models at https://huggingface.co/unsloth

    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen3-VL-8B-Instruct",
        load_in_4bit=True,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    return model, tokenizer

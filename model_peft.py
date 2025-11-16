from unsloth import FastVisionModel  # FastLanguageModel for LLMs
import torch
from base_model import get_base_model


def get_peft_model():
    '''
    把原始模型包装成 LoRA 模型，让它支持低秩适配器训练，同时冻结主干、只训练 LoRA 参数
    :return:
    '''
    model, tokenizer = get_base_model()

    # add LoRA adapters for parameter efficient finetuning -
    # this allows us to only efficiently train 1% of all parameters.
    # 在 Unsloth 0.5.2+  from_pretrained() 的实现里已经自动帮你把 LoRA 加好了（内部默认调了一次 get_peft_model）
    if not hasattr(model, "peft_config"):
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,  # False if not finetuning vision layers
            finetune_language_layers=True,  # False if not finetuning language layers
            finetune_attention_modules=True,  # False if not finetuning attention layers
            finetune_mlp_modules=True,  # False if not finetuning MLP layers

            r=16,  # The larger, the higher the accuracy, but might overfit
            lora_alpha=16,  # Recommended alpha == r at least
            lora_dropout=0,
            bias="none",
            random_state=3407,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

    return model

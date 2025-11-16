from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from unsloth import FastVisionModel
from model_peft import get_peft_model

from datas import get_latex_ocr_data, convert_to_conversation


def sft_model_train():
    # laod data
    dataset = get_latex_ocr_data()

    # VisionModel train_dataset
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    # load model
    model, tokenizer = get_peft_model()
    FastVisionModel.for_training(model)  # Enable for training!

    # UnslothVisionDataCollator which will help in our vision finetuning setup.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
        train_dataset=train_dataset,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=30,
            # num_train_epochs = 1, # Set this instead of max_steps for full training runs
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            report_to="none",  # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=2048,
        ),
    )

    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # training model
    trainer_stats = trainer.train()


    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # save lora model
    '''
    只保存 LoRA 适配器权重（几十~几百 MB）。
    不合并原模型，文件里只有 adapter_config.json + adapter_model.safetensors。
    加载时必须把原模型也一起加载，再 PeftModel.from_pretrained(base, adapter_dir)
    '''
    model.save_pretrained("lora_model")  # Local saving
    tokenizer.save_pretrained("lora_model")
    # model.push_to_hub("your_name/lora_model", token = "...") # Online saving
    # tokenizer.push_to_hub("your_name/lora_model", token = "...") # Online saving

    # Saving to float16 for VLLM Save locally to 16 bit
    '''
    LoRA 权重合并回主干模型，保存完整模型（8B 模型就是 15 GB 左右）。
    生成的是标准 Hugging Face 格式：config.json + model-00001-of-xxx.safetensors …
    加载时不再需要 PEFT，直接用 AutoModelForVisionSeq2Seq.from_pretrained("merged_dir")
    '''
    model.save_pretrained_merged("unsloth_finetune", tokenizer, )
    # To export and save to your Hugging Face account
    # model.push_to_hub_merged("YOUR_USERNAME/unsloth_finetune", tokenizer, token="PUT_HERE")

    '''
    GGUF / llama.cpp Conversion
    '''
    # Save to 8bit Q8_0
    if False: model.save_pretrained_gguf("unsloth_finetune", tokenizer, )
    # Remember to go to https://huggingface.co/settings/tokens for a token!
    # And change hf to your username!
    if False: model.push_to_hub_gguf("hf/unsloth_finetune", tokenizer, token="")

    # Save to 16bit GGUF
    if False: model.save_pretrained_gguf("unsloth_finetune", tokenizer, quantization_method="f16")
    if False: model.push_to_hub_gguf("hf/unsloth_finetune", tokenizer, quantization_method="f16", token="")

    # Save to q4_k_m GGUF
    if False: model.save_pretrained_gguf("unsloth_finetune", tokenizer, quantization_method="q4_k_m")
    if False: model.push_to_hub_gguf("hf/unsloth_finetune", tokenizer, quantization_method="q4_k_m", token="")

    # Save to multiple GGUF options - much faster if you want multiple!
    if False:
        model.push_to_hub_gguf(
            "hf/unsloth_finetune",  # Change hf to your username!
            tokenizer,
            quantization_method=["q4_k_m", "q8_0", "q5_k_m", ],
            token="",
        )
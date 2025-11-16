from unsloth import FastVisionModel
from transformers import TextStreamer
from datas import get_latex_ocr_data

def model_inference(model_path: str, image):

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_path,  # "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=True,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)

    # image = dataset[2]["image"]
    instruction = "Write the LaTeX representation for this image."

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")


    text_streamer = TextStreamer(tokenizer, skip_prompt=True)
    _ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=128,
                       use_cache=True, temperature=1.5, min_p=0.1)


if __name__ == '__main__':
    model_path = 'YOUR MODEL'
    dataset = get_latex_ocr_data()
    image = dataset[2]["image"]
    model_inference(model_path=model_path, image=image)


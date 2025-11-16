from datasets import load_dataset
from IPython.display import display, Math, Latex

def get_latex_ocr_data():
    dataset = load_dataset("unsloth/LaTeX_OCR", split = "train")  # 9:1
    return dataset

def display_latex():
    dataset = get_latex_ocr_data()
    latex = dataset[2]["text"]
    display(Math(latex))

def convert_to_conversation(sample):
    '''
    Let's convert the dataset into the "correct" format for finetuning:
    :param sample:
    :return:
    '''

    instruction = "Write the LaTeX representation for this image."
    conversation = [
        {"role": "user",
         "content": [
             {"type": "text", "text": instruction},
             {"type": "image", "image": sample["image"]}]
         },
        {"role": "assistant",
         "content": [
             {"type": "text", "text": sample["text"]}]
         },
    ]
    return {"messages": conversation}
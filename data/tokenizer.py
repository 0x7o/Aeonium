from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

dataset1 = load_dataset("IlyaGusev/rulm")["train"]
dataset2 = load_dataset("code_search_net", "all")

data = {"text": []}

for data in tqdm(dataset1["train"]):
    data["text"].append(data["text"])

for data in tqdm(dataset2["test"]):
    data["text"].append(data["whole_func_string"])

dataset_cc = Dataset.from_dict(data)
dataset_cc = dataset_cc.shuffle()

tokenizer = AutoTokenizer.from_pretrained("chargoddard/llama3-42b-v0")

batch_size = 1024


def batch_iterator():
    for i in range(0, len(dataset_cc), batch_size):
        yield dataset_cc[i : i + batch_size]["text"]


new_tokenizer = tokenizer.train_new_from_iterator(
    batch_iterator(),
    vocab_size=128000,
    new_special_tokens=[
        "<|end_of_text|>",
        "<|begin_of_text|>",
        "<|im_end|>",
        "<|im_start|>",
    ],
)

new_tokenizer.push_to_hub("aeonium/Aeonium-v1-Base-7B")

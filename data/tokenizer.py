from datasets import load_dataset
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast

dataset1 = load_dataset("IlyaGusev/rulm")["train"]
dataset2 = load_dataset("code_search_net", "all")["test"]

tokenizer = ByteLevelBPETokenizer()

batch_size = 1000


def batch_iterator():
    for i in range(0, len(dataset1), batch_size):
        yield dataset1[i : i + batch_size]["text"]

    for i in range(0, len(dataset2), batch_size):
        yield dataset2[i : i + batch_size]["whole_func_string"]


new_tokenizer = tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=128000,
    special_tokens=[
        "<|end_of_text|>",
        "<|begin_of_text|>",
        "<|im_end|>",
        "<|im_start|>",
    ],
)

tokenizer = PreTrainedTokenizerFast(new_tokenizer)
tokenizer.push_to_hub("aeonium/Aeonium-v1-Base-7B")
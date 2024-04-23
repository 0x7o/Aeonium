from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

dataset1 = load_dataset("IlyaGusev/rulm")
dataset1 = dataset1.remove_columns([
        col for col in dataset1.column_names if col != "text"
    ])

tokenizer = ByteLevelBPETokenizer()

batch_size = 1000


def batch_iterator():
    for i in range(0, len(dataset1), batch_size):
        yield dataset1["train"][i : i + batch_size]["text"]


new_tokenizer = tokenizer.train_from_iterator(
    batch_iterator(),
    vocab_size=128000,
    special_tokens=[
        "<|end_of_text|>",
        "<|im_end|>",
        "<|im_start|>",
    ],
)

tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=new_tokenizer,
    bos_token="<|end_of_text|>",
    eos_token="<|end_of_text|>",
)
tokenizer.push_to_hub("aeonium/Aeonium-v1-Base-7B")

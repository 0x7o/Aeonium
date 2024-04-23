from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

dataset1 = load_dataset("IlyaGusev/rulm")["train"]
dataset1 = dataset1.remove_columns([
        col for col in dataset1.column_names if col != "text"
    ])

tokenizer = ByteLevelBPETokenizer()

batch_size = 1000


def batch_iterator(dataset, batch_size=1_000):
    for batch in dataset.iter(batch_size=batch_size):
        yield batch["text"]


new_tokenizer = tokenizer.train_from_iterator(
    batch_iterator(dataset1),
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

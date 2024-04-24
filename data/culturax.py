from datasets import Dataset
from transformers import AutoTokenizer
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import pyarrow as pa
import requests
import os

hf_token = os.getenv("HF_TOKEN")
tokenizer = AutoTokenizer.from_pretrained("aeonium/Aeonium-v1-Base-7B")


def download_file(file_path: str, url: str):
    response = requests.get(
        url, headers={"Authorization": f"Bearer {hf_token}"}, stream=True
    )

    if os.path.exists(file_path):
        return

    with open(file_path, "wb") as file:
        for chunk in tqdm(
            response.iter_content(chunk_size=128),
            total=int(response.headers.get("content-length", 0)) / 128,
        ):
            file.write(chunk)


def process_text(text: str):
    return tokenizer.encode_plus(text)


def save_pickle(data, file_path):
    table = pa.Table.from_arrays([data], names=["tokens"])
    pq.write_table(table, file_path)


def tokenization(example):
    return tokenizer(example["text"])


def main(output_dir: str, batch_size: int):
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(0, 1535), desc="Processing files..."):
        file_path = f"{output_dir}/ru_part_{str(i).zfill(5)}.parquet"
        url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/ru/ru_part_{str(i).zfill(5)}.parquet?download=true"
        download_file(file_path, url)

        table = pq.read_table(file_path)
        dataset = Dataset.from_pandas(table.to_pandas())

        dataset.map(tokenization, batched=True, batch_size=batch_size).to_parquet(
            f"{output_dir}/ru_part_{str(i).zfill(5)}.parquet"
        )

        os.remove(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1000)
    args = parser.parse_args()

    main(args.output_dir, args.batch_size)

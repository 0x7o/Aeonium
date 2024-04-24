from transformers import AutoTokenizer
from multiprocessing import Pool
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import requests
import pickle
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


def parquet_iterator(file_path: str):
    table = pq.read_table(file_path)
    for row in table[0]:
        yield str(row)


def process_text(text: str):
    return tokenizer.encode_plus(text)


def save_pickle(data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def main(output_dir: str, num_workers: int):
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, 1535):
        file_path = f"{output_dir}/ru_part_{str(i).zfill(5)}.parquet"
        url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/ru/ru_part_{str(i).zfill(5)}.parquet?download=true"
        download_file(file_path, url)

        with Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_text, parquet_iterator(file_path))))

        save_pickle(results, f"{output_dir}/ru_part_{str(i).zfill(5)}.pkl")
        os.remove(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args.output_dir, args.num_workers)

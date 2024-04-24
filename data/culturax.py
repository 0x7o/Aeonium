from transformers import AutoTokenizer
from multiprocessing import Pool
import pyarrow.parquet as pq
from tqdm import tqdm
import argparse
import requests
import pickle
import os

hf_token = os.getenv("HF_TOKEN")


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


def parquet_iterator(table):
    for row in table[0]:
        yield str(row)


def save_pickle(data, file_path):
    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def process_batch(args):
    batch, tokenizer = args
    batch_text = [str(row) for row in batch]
    return tokenizer.batch_encode_plus(batch_text)['input_ids']


def process_file(file_path, output_dir, num_workers):
    table = pq.read_table(file_path)
    tokenizer = AutoTokenizer.from_pretrained("aeonium/Aeonium-v1-Base-7B")

    batch_size = len(table) // num_workers
    batches = [table[i:i + batch_size] for i in range(0, len(table), batch_size)]

    with Pool(num_workers) as pool:
        args = [(batch, tokenizer) for batch in batches]
        results = list(tqdm(pool.imap_unordered(process_batch, args), total=len(args)))

    flattened_results = [item for sublist in results for item in sublist]
    save_pickle(flattened_results, f"{output_dir}/{os.path.basename(file_path).split('.')[0]}.pkl")
    os.remove(file_path)


def main(output_dir: str, num_workers: int):
    os.makedirs(output_dir, exist_ok=True)

    for i in tqdm(range(0, 1535), desc="Processing files..."):
        file_path = f"{output_dir}/ru_part_{str(i).zfill(5)}.parquet"
        url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/ru/ru_part_{str(i).zfill(5)}.parquet?download=true"
        download_file(file_path, url)
        process_file(file_path, output_dir, num_workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    main(args.output_dir, args.num_workers)

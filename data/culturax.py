import pyarrow.parquet as pq
from tqdm import tqdm
import requests
import mlxu
import json
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


def main():
    for i in tqdm(range(0, 1535), desc="Processing files..."):
        file_path = f"ru_part_{str(i).zfill(5)}.parquet"
        url = f"https://huggingface.co/datasets/uonlp/CulturaX/resolve/main/ru/ru_part_{str(i).zfill(5)}.parquet?download=true"
        download_file(file_path, url)

        table = pq.read_table(file_path)

        with mlxu.utils.open_file(
            f"gs://aeonium-checkpoints/culturax/ru_part_{str(i).zfill(5)}.json", "w"
        ) as file:
            batch = ""
            for item in table[0]:
                text = str(item)
                file.write(json.dumps({"text": text}) + "\n")
            file.write(batch)

        os.remove(file_path)


if __name__ == "__main__":
    main()

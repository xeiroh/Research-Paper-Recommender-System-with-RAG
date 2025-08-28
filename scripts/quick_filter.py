import pandas as pd
import numpy as np
from app import settings
import os

DATA = settings.data_dir
FILE = os.path.join(DATA, "paperswithcode.json")

data = pd.read_json(FILE)
data.reset_index(drop=True, inplace=True)
data["uid"] = data.index
data.drop_duplicates(subset=["arxiv_id", "abstract"], inplace=True)

data["title"] = data["title"].fillna("").astype(str).str.strip().replace("", np.nan)
data["abstract"] = data["abstract"].fillna("").astype(str).str.strip().replace("", np.nan)

data.dropna(subset=["title", "abstract"], inplace=True)
data["content"] = data["title"] + " " + data["abstract"]

NEW_FILE = os.path.join(DATA, "pwc_filtered.jsonl")
data.to_json(NEW_FILE, orient="records", lines=True)
data.to_parquet(os.path.join(DATA, "paperswithcode.parquet"), index=False)
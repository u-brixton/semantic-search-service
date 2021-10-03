import json
from pathlib import Path
from typing import List, Literal, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from semantic_search import get_embedder, load_faiss_index, retrieve_similar_phrases

DATA_CONFIG_PATH = "artifacts/data_config.json"
data_config = json.load(open(DATA_CONFIG_PATH, "r"))
allowed_datasets_list = list(data_config["datasets"])

model = get_embedder(data_config["model"]["path"])
faiss_indices = dict()
text_datasets = dict()

for source_name in data_config["datasets"]:
    faiss_indices[source_name] = load_faiss_index(
        data_config["datasets"][source_name]["index_path"]
    )
    text_datasets[source_name] = pd.read_csv(
        data_config["datasets"][source_name]["text_path"]
    )

# TODO: add assert to check indices

app = FastAPI(
    title="Semantic search service",
    description="Retrieve phrases similar to a query",
    version="1.0.0",
)


class Input(BaseModel):
    query_phrase: str
    top_n: Optional[int] = 20
    dataset_name: str  # Literal[tuple(allowed_datasets_list)]


# class Output(BaseModel):
#     query_phrase: str
#     top_n: int
#     dataset_name:
#     result: List[str]


@app.get("/")
def home():
    return {
        "message": "Visit the endpoint: /api/v1/get_similar_phrases to perform OCR. Visit the endpoint /docs for documentation"
    }


# TODO: add method to get all possible sources
# TODO: add async to functions


@app.post("/api/v1/get_similar_phrases/")
def get_similar_phrases(input: Input):  # , response_model=Output):
    similar_phrases = retrieve_similar_phrases(
        input.query_phrase,
        faiss_indices[input.dataset_name],
        text_datasets[input.dataset_name],
        model,
        top_n=input.top_n,
    )

    output = dict()
    # REFACTOR ME
    output["query_phrase"] = input.query_phrase
    output["top_n"] = input.top_n
    output["use_dataset"] = input.use_dataset
    output["result"] = similar_phrases

    return output

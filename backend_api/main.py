import asyncio
import json
from pathlib import Path
from typing import List, Dict, Literal, Optional

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

app = FastAPI(
    title="Semantic search service",
    description="Retrieve phrases similar to a query",
    version="1.0.0",
)


class Input(BaseModel):
    query_phrase: str
    top_n: Optional[int] = 20
    dataset_name: Literal[tuple(allowed_datasets_list)] = allowed_datasets_list[0]


class Output(BaseModel):
    query_phrase: str
    top_n: int
    dataset_name: Literal[tuple(allowed_datasets_list)]
    result: List[str]


@app.get("/")
async def home():
    return {
        "message": "Visit the endpoint: /api/v1/get_similar_phrases to perform OCR. Visit the endpoint /docs for documentation"
    }


@app.post("/api/v1/get_available_datasets/")
async def get_available_datasets() -> Dict[str, List[str]]:
    """
    Return a list of datasets name, which we can use to find similar phrases
    """
    return {"allowed_datasets": allowed_datasets_list}


@app.post("/api/v1/get_similar_phrases/")
async def get_similar_phrases(input: Input) -> Output:
    """
    Returns a list of similar phrases given an input phrase, faiss index, model and texts for faiss model
    """
    similar_phrases = retrieve_similar_phrases(
        input.query_phrase,
        faiss_indices[input.dataset_name],
        text_datasets[input.dataset_name],
        model,
        top_n=input.top_n,
        text_column="line",
    )

    output = input.dict()
    output["result"] = similar_phrases

    return output

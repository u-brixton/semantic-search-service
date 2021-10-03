import pickle
from typing import Union

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def get_embedder(model_path):
    return SentenceTransformer(str(model_path))


def load_faiss_index(index_path):
    with open(index_path, "rb") as fin:
        index = faiss.deserialize_index(pickle.load(fin))
    return index


def save_faiss_index(index_path):
    with open(index_path, "wb") as fin:
        index = faiss.serialize_index(pickle.load(fin))
    return index


def retrieve_similar_phrases(
    query_phrase, faiss_index, text_dataset, model, top_n=20, text_column="text"
):
    query_emb = np.expand_dims(model.encode(query_phrase), 0)
    _, ids = faiss_index.search(x=query_emb, k=top_n)
    result = text_dataset.iloc[ids[0], :][text_column].tolist()
    return result

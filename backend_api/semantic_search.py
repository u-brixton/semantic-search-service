import pickle
from pathlib import Path
from typing import List, Union

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def get_embedder(model_path: Union[Path, str]) -> SentenceTransformer:
    """
    :param model_path: path to SentenceTransformer model weights and parameters
    :returns: a loaded Sentence Transformer Model
    """
    return SentenceTransformer(str(model_path))


def load_faiss_index(
    index_path: Union[Path, str],
    nprobe: Union[int, None] = None,
):
    """
    Loads faiss index into memory
    :param index_path: path to serialized faiss weights
    """
    with open(index_path, "rb") as fin:
        index = faiss.deserialize_index(pickle.load(fin))
    if nprobe is not None:
        index.nprobe = nprobe
    return index


def save_faiss_index(faiss_index, index_path: Union[Path, str]) -> None:
    """
    Serializes faiss index and dumps it to disk
    :param faiss_index: faiss index to save
    :param index_path: path where to serialize faiss index
    """
    with open(index_path, "wb") as fout:
        pickle.dump(faiss.serialize_index(faiss_index), fout)


def retrieve_similar_phrases(
    query_phrase: str,
    faiss_index,
    text_dataset: pd.DataFrame,
    model: SentenceTransformer,
    top_n: int = 20,
    text_column: str = "line",
) -> List[str]:
    """
    Finds nearest neighbours for query phrase given a set of phrases and a faiss index
    :param query_phrase: text phrase for which we should find similar phrases
    :param faiss_index: faiss index to be used for search
    :param text_dataset: pandas DataFrame with phrases
    :param model: SentenceTransformer model to encode query_phase, should be the same that used in faiss index
    :param top_n: number of similar phrases to find
    :param text_column: which column to use in text_dataset to find similar phrases
    """
    query_emb = np.expand_dims(model.encode(query_phrase), 0)
    _, ids = faiss_index.search(x=query_emb, k=top_n)
    similar_phrases = text_dataset.iloc[ids[0], :][text_column].tolist()
    return similar_phrases

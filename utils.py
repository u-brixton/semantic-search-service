import json
import re

import numpy as np
import pandas as pd
from IPython.display import display

pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.max_columns", 10)
import gzip
import math
import os
import pickle
import shutil
from pathlib import Path
from typing import Dict, Union

import faiss
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import tqdm


def unify_dataframe_and_get_stats(
    df: pd.DataFrame, ds_name: str, new_text_col: str = "line"
):
    assert df.shape[1] == 1, "Should be dataframe with one text column"
    df.columns = [new_text_col]
    df.dropna(inplace=True)
    df[new_text_col] = df[new_text_col].str.strip()
    df = df[df[new_text_col] != ""]
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["length"] = df[new_text_col].apply(len)
    print(f"Dataset {ds_name} has {len(df)} lines")
    print(
        f'Max line length is {max(df["length"])}, min line length is {df["length"].min()}'
    )
    print(
        f'Median line length is {df["length"].median()}, 99 pct is {df["length"].quantile(0.9)}'
    )
    display(df.head(3))
    return df


def preproc_df(
    df: pd.DataFrame,
    remove_curly_brackets: bool = True,
    min_length: int = 2,
    max_length: int = 150,
    text_col: str = "line",
    length_col: str = "length",
):
    """
    Removes phrases with very long or very short lengths, removes curly brackets and their content
    :param df: input dataframe
    :param remove_curly_brackets: whether to remove brackets or not
    """
    pattern = re.compile(r"\(.*\)")
    if remove_curly_brackets:
        df[text_col] = df[text_col].apply(lambda x: pattern.sub("", x))
        df = df.drop_duplicates(text_col)
        df = df[df[text_col] != ""]
        df["length"] = df[text_col].apply(len)

    df = df[df[length_col] >= min_length]
    df = df[df[length_col] <= max_length]

    df = df.reset_index(drop=True)

    return df


def calculate_and_save_faiss_index(
    df: pd.DataFrame,
    model: SentenceTransformer,
    index_save_path: Union[str, Path, None],
    faiss_nlist: int = None,
    faiss_nprobe: int = 50,
    text_col: str = "line",
):
    """
    Builds, trains and saves to disk faiss Inverted File index with DotProduct distance metric
    :param df: pandas DataFrame with sentences to embed
    :param model: Loaded sentence transformer model with a pooling to embed sentences
    :param index_save_path: path to save trained index
    :param faiss_nlist: number of partitions we like our index to have
    :param faiss_nprobe: number of nearby cells to search when searching using index. Larger -> less speed, but higher quality
    :param text_col: text column name in df
    """
    embs = model.encode(df[text_col].tolist())
    assert embs.shape[0] == len(df)
    # we will use L2 distance which is equivalent to cosine similarity if all vectors are normalized
    assert np.allclose(np.linalg.norm(embs, axis=1), 1)
    D = embs.shape[1]

    quantizer = faiss.IndexFlatIP(D)

    # Number of clusters used for faiss
    # Select a value 4*sqrt(N) to 16*sqrt(N) - https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
    if faiss_nlist is None:
        faiss_nlist = int(math.sqrt(len(df)) * 8)
    cpu_index = faiss.IndexIVFFlat(
        quantizer, D, faiss_nlist, faiss.METRIC_INNER_PRODUCT
    )
    cpu_index.nprobe = faiss_nprobe
    cpu_index.train(embs)
    cpu_index.add(embs)

    if index_save_path is not None:
        with open(index_save_path, "wb") as fout:
            pickle.dump(faiss.serialize_index(cpu_index), fout)

    return cpu_index


def prepare_artifacts(
    dfs: Dict[str, pd.DataFrame],
    save_path: Union[str, Path],
    pretrained_model_path: Union[str, Path],
    text_col: str = "line",
    model_fd: str = "model",
    indices_fd: str = "indices",
    texts_fd: str = "texts",
):
    """
    Saved faiss indices/text data/data config given list of dataframes and their names and a embedder model
    :param dfs: Dict with dataframes and their respective names
    :param save_path: Path where to save text, indices and data_config.json
    :param pretrained_model_path: Path to saved sentence transformer model with a pooling to embed sentences
    :param text_col: Text column name in df
    :param model_fd: Name of the folder inside a save_path where model weights will be copied
    :param indices_fd: Name of the folder inside a save_path where faiss indices will be saved
    :param texts_fd: Name of the folder inside a save_path where phrase dataframes will be saved
    """
    save_path = Path(save_path)
    save_path.mkdir(exist_ok=True)
    (save_path / model_fd).mkdir(exist_ok=True)
    (save_path / texts_fd).mkdir(exist_ok=True)
    (save_path / indices_fd).mkdir(exist_ok=True)

    indices = dict()
    data_config = dict()
    model_name = Path(pretrained_model_path).name
    data_config["model"] = dict()
    data_config["datasets"] = dict()
    data_config["model"]["path"] = str(save_path / model_fd)

    print("Copying a model")

    copy_and_overwrite(pretrained_model_path, data_config["model"]["path"])
    model = SentenceTransformer(data_config["model"]["path"])

    for i, (ds_name, df) in tqdm(enumerate(dfs.items()), total=len(dfs)):
        print(f"Processing {ds_name} dataset: ({i + 1}/{len(dfs)})")
        data_config["datasets"][ds_name] = dict()

        df_save_path = save_path / texts_fd / f"texts_{ds_name}.csv"
        df.to_csv(df_save_path)

        print(f"Calculating embeddings and saving them to faiss index")
        index_save_path = save_path / indices_fd / f"faiss_index_{ds_name}.pkl"
        indices[ds_name] = calculate_and_save_faiss_index(df, model, index_save_path)

        data_config["datasets"][ds_name]["index_path"] = str(index_save_path)
        data_config["datasets"][ds_name]["text_path"] = str(df_save_path)

    with open(save_path / "data_config.json", "w") as fout:
        json.dump(data_config, fout, ensure_ascii=False, indent=4)

    return model, data_config, indices


def copy_and_overwrite(from_path: Union[str, Path], to_path: Union[str, Path]):
    """
    Copies contents from one folder to another
    """
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

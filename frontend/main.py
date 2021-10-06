import pandas as pd
import requests
import streamlit as st

BACKEND_ADDRESS = "http://backend-api"
BACKEND_PORT = 8000

st.sidebar.markdown("**Inputs**")
allowed_datasets = requests.post(
    f"{BACKEND_ADDRESS}:{BACKEND_PORT}/api/v1/get_available_datasets/"
).json()["allowed_datasets"]

st.title("Semantic search web app")
st.write(
    """Retrieve similar phrases from 3 different datasets.
         This examples uses FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)

form_input = st.sidebar.text_area("Search box", "Happy New Year")
dataset_name = st.sidebar.selectbox(
    "Choose the dataset name for similar phrases search", allowed_datasets
)
top_n = st.sidebar.slider("Number of neighbours", 1, 100, 20)


if st.sidebar.button("Retrieve similar phrases from a dataset"):
    post_params = {
        "query_phrase": form_input,
        "top_n": top_n,
        "dataset_name": dataset_name,
    }
    res = requests.post(
        f"{BACKEND_ADDRESS}:{BACKEND_PORT}/api/v1/get_similar_phrases/",
        json=post_params,
    ).json()

    st.table(pd.DataFrame(res["similar_phrases"], columns=["similar_phrases"]))

# Semantic search webapp

## About
This is a semantic search webapp with a frontend interface on Streamlit and backend interface on FastAPI

You can choose one of the 3 datasets, top_n of neighbours and find most similar phrases from the chosen dataset

Those datasets are: 
- main dataset with informal phrases 
- quotes from "Friends" TV series
- quotes from "Futurama" TV series

After you build an app, API docs can be found at :8000/docs, main API endpoint is :8000/api/v1/get_similar_phrases/

Visual frontend interface built on streamlit can be found at :8051

## Where to start

Build an app using commands in the section "how to build" or check the data preparation notebooks in :

notebooks/semantic-search_save-artifacts.ipynb

## How to download artifacts

Artifacts will be downloaded automatically when you build the docker containers, but you can control it with an ARG DOWNLOAD_DATA for backend_api build

You can download artifacts manually (embedder model, faiss indices and texts) using 
``` bash
artifacts/download_artifacts.sh
```
Folders /model, /texts, /indices will be created, artifacts will be downloaded and data_config.json will appear inside of artifacts folder after you use the script

data_config.json will later be used in the app, as paths to all artifacts are saved there

## How to change embedder model / texts used to mine similar phrases
To prepare your own artifacts you can follow the tutorial notebook in notebooks/semantic-search_save-artifacts.ipynb

## How to build
To run locally use:

``` bash
docker-compose build
docker-compose up
```

To run in cloud you can use:
- docker ecs compose (but to allocate enough resources): https://docs.docker.com/cloud/ecs-integration/
- AWS Elastic Beanstalk with Docker Amazon Linux 2 Platform (but to allocate enough resources): https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/docker-multicontainer-migration.html
- Heroku DockHero

## Screenshots
![Screenshot for a demo](demo_screenshot.png?raw=true)
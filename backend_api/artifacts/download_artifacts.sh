#!/bin/bash -ex

# we download model and data together, as indices can be used only with model with which they were created

#BASE_URL="https://semantic-search-oct21.s3.eu-west-1.amazonaws.com/artifacts.zip"
BASE_URL="https://storage.yandexcloud.net/semantic-search/artifacts.zip"
ZIP_NAME="artifacts.zip"
DIRECTORY=`dirname $0`

if test $1 -eq 0; then
    echo "Not downloading data as input flag is 0"
else 
    echo "Downloading artifacts: model, faiss precalculated indices and text datasets"
    curl -ljo $ZIP_NAME $BASE_URL
    unzip $ZIP_NAME -d $DIRECTORY
    rm $ZIP_NAME
fi

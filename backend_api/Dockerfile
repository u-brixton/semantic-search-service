FROM tiangolo/uvicorn-gunicorn:python3.8

ARG DOWNLOAD_DATA=0

RUN mkdir /backend

COPY requirements.txt /backend
WORKDIR /backend
RUN pip install -r requirements.txt


COPY . /backend
RUN chmod +x artifacts/download_artifacts.sh && artifacts/download_artifacts.sh ${DOWNLOAD_DATA}

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
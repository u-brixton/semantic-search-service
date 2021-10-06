FROM tiangolo/uvicorn-gunicorn:python3.8

ARG DOWNLOAD_DATA=1

RUN mkdir /backend

COPY requirements.txt /backend
WORKDIR /backend
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir http://download.pytorch.org/whl/cpu/torch-1.8.1%2Bcpu-cp38-cp38-linux_x86_64.whl \
    pip install --no-cache-dir -r requirements.txt

COPY . /backend
RUN chmod +x artifacts/download_artifacts.sh && artifacts/download_artifacts.sh ${DOWNLOAD_DATA}

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
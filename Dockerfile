FROM python:3.10-slim

WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY serving/ ./serving/
COPY config.yaml .
COPY mlruns/ ./mlruns/

ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5001

EXPOSE 5000

CMD ["python", "serving/main.py"]

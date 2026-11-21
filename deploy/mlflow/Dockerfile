# MLflow tracking server with the Postgres driver and the S3 client.
# Same MLflow version as requirements.txt resolves to, so clients and server match.
# The compose file names the driver in the URI (postgresql+psycopg2://), because
# newer SQLAlchemy defaults plain postgresql:// to psycopg 3.
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    "mlflow==3.18.0" \
    "psycopg2-binary>=2.9,<3" \
    "boto3>=1.34,<2"

WORKDIR /app
COPY create_bucket.py .

EXPOSE 5000

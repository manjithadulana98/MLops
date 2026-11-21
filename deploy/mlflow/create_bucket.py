"""Create the artifact bucket if it doesn't exist yet (runs before the MLflow server).

Retries for a while because the object store may still be starting.
"""
import os
import sys
import time

import boto3
from botocore.exceptions import ClientError, EndpointConnectionError

bucket = os.environ["ARTIFACT_BUCKET"]
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])

for attempt in range(1, 31):
    try:
        existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
        if bucket not in existing:
            s3.create_bucket(Bucket=bucket)
            print(f"Created bucket {bucket}")
        else:
            print(f"Bucket {bucket} exists")
        sys.exit(0)
    except (ClientError, EndpointConnectionError) as exc:
        print(f"Object store not ready (attempt {attempt}/30): {exc}")
        time.sleep(2)

sys.exit("Object store never became ready")

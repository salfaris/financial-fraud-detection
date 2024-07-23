#!/bin/bash

BUCKET_NAME="transactions-fraud-datasets"
BUCKET_LOCATION="europe-west2"

gcloud storage buckets create gs://$BUCKET_NAME --location=$BUCKET_LOCATION
gcloud storage cp -r "./datasets/01_raw" "gs://$BUCKET_NAME/01_raw"
gcloud storage cp -r "./datasets/02_staged" "gs://$BUCKET_NAME/02_staged"
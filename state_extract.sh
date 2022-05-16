#!/bin/bash

# Parse args if specified.
if [ $# -lt 1 ]; then
  echo "Usage: $0 [state-code] (GCS bucket name and path)"
  echo
  echo "Example: $0 CT gs://test-bucket/extract_data"
  echo "Example: $0 CA gs://test-bucket/extract_data/CA/"
  echo "Example: $0 TX gs://test-bucket/extract_data/"
  exit 1
else
  STATE="${1}"
fi

if [ $# -eq 1 ]; then
  gcs_bucket="gs://default-bucket/path/"
else
  gcs_bucket="${2}"
fi

echo $STATE
echo $gcs_bucket

# step 1 : Generate pickle file for step 2
./run.py data update --state $STATE

# step 2 : extract API data for the state
./run.py api generate-api-v2 --state $STATE output -o output/api

today_date=$(date '+%Y-%m-%d')

gsutil cp output/api/states.timeseries.csv phc-can-internal/state/${today_date}/
gsutil cp output/api/states.timeseries.csv phc-can-internal/county/${today_date}/

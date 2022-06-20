#!/usr/bin/env bash
set -euo pipefail

warn(){ >&2 echo "$@"; }

main() {
  : "${DATA_OUTPUT_DIR?Expected}"
  : "${GCS_BUCKET_PREFIX?Expected}"
  : "${GCS_STORAGE_BUCKET?Expected}"

  local base_dir output_dir today_date upload_path

  base_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
  today_date=$(date -I)
  upload_path="gs://${GCS_STORAGE_BUCKET}/${GCS_BUCKET_PREFIX}/${today_date}/"

  warn "Data Output Dir ${DATA_OUTPUT_DIR}"
  warn "GCS upload path: ${upload_path}"

  warn "Working dir $base_dir"
  output_dir="${base_dir}/output"
  mkdir -p "$output_dir" "$DATA_OUTPUT_DIR"

  warn "Fetching nyt_anomalies.csv"
  curl -sf https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/anomalies.csv \
    --output "$base_dir/data/nyt_anomalies.csv"

  warn "Starting LFS Pull and Prune"
  git -C "$base_dir" lfs pull
  git -C "$base_dir" lfs prune
  warn "Finished LFS Pull and Prune"

  warn "Updating datasets"
  time "$base_dir/run.py" data update --refresh-datasets
  # approx 30m
  warn "Finished updating datasets"

  warn "pyseir build-all"
  time pyseir build-all
  # approx 16m
  warn "Finished pyseir build-all"

  warn "Generating api-v2 data"
  time "$base_dir/run.py" api generate-api-v2 "$output_dir" -o "$DATA_OUTPUT_DIR"
  # approx 60m
  # use ~21GB
  warn "Finished generating api-v2 data in $DATA_OUTPUT_DIR"

  warn "Publishing to $upload_path"
  gsutil -m cp -r "$DATA_OUTPUT_DIR" "$upload_path"
  warn "Finished publishing to $upload_path"
  # gsutil -m cp -r /data/api-results gs://phc-sbx-prefect/phc-can-internal/2022-06-21/
}

main "$@"

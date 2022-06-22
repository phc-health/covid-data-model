#!/usr/bin/env bash
set -euo pipefail

warn(){ >&2 echo "$@"; }

log_step() {
  : "${step?expected}"
  : "${prev_step:=''}"

  if [[ $prev_step != "$step" ]]; then
    warn ""
    warn "[START] $step @ $(date --rfc-3339=seconds)"
    prev_step=$step
  else
    warn "[END] $step @ $(date --rfc-3339=seconds)"
  fi
}

main() {
  : "${DATA_OUTPUT_DIR?Expected}"
  : "${GCS_BUCKET_PREFIX?Expected}"
  : "${GCS_STORAGE_BUCKET?Expected}"
  : "${UPLOAD_FILE_FILTER?Expected}"

  local base_dir output_dir today_date upload_path step prev_step

  warn "Data Output Dir: ${DATA_OUTPUT_DIR}"

  base_dir=$(realpath "$(dirname "${BASH_SOURCE[0]}")")
  warn "Working Dir: $base_dir"

  today_date=$(date -I)
  upload_path="gs://${GCS_STORAGE_BUCKET}/${GCS_BUCKET_PREFIX}/${today_date}/"
  warn "GCS Upload Path: ${upload_path}"

  output_dir="${base_dir}/output"
  mkdir -p "$output_dir" "$DATA_OUTPUT_DIR"

  step="Fetching nyt_anomalies.csv"
  time {
    log_step
    curl -sf https://raw.githubusercontent.com/nytimes/covid-19-data/master/rolling-averages/anomalies.csv \
      --output "$base_dir/data/nyt_anomalies.csv"
    log_step
  }

  step="LFS Pull and Prune"
  time {
    log_step
    git -C "$base_dir" lfs pull
    git -C "$base_dir" lfs prune
    log_step
  }

  step="Updating datasets"
  time {
    log_step
    "$base_dir/run.py" data update --refresh-datasets 2>&1
    # approx 73m
    log_step
  }

  step="pyseir build-all"
  time {
    log_step
    pyseir build-all 2>&1
    # approx 15m
    log_step
  }

  step="Generating api-v2 data in $DATA_OUTPUT_DIR"
  time {
    log_step
    "$base_dir/run.py" api generate-api-v2 "$output_dir" -o "$DATA_OUTPUT_DIR" 2>&1
    # approx 65m
    # started at 8gb
    # used ~21GB at end
    log_step
  }

  step="Publishing to $upload_path"
  time {
    log_step
    find "$DATA_OUTPUT_DIR" -regextype egrep -regex "$UPLOAD_FILE_FILTER" | gsutil -m cp -I "$upload_path"
    log_step
  }
}

main "$@"

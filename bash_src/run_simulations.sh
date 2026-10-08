#!/bin/bash

config_dir="./configs_to_run"

if [[ ! -d "$config_dir" ]]; then
  echo "Config directory not found: $config_dir" >&2
  exit 1
fi

shopt -s nullglob
configs=("$config_dir"/*.json)
shopt -u nullglob

if (( ${#configs[@]} == 0 )); then
  echo "No config files found in: $config_dir" >&2
  exit 1
fi

for config_path in "${configs[@]}"; do
  echo "Found config file: $config_path"
done

for config_path in "${configs[@]}"; do
  echo "Running ./bin/main_buryn $config_path"
  ./bin/main_buryn "$config_path"
  
done
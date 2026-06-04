#!/bin/bash

# Configuration
MAX_INSTANCES=8
CPUS_PER_INSTANCE=14
GPUS_PER_INSTANCE=1

# Get the total number of CPUs and GPUs available
TOTAL_CPUS=$(nproc)
TOTAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Check if an instance number is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <instance_number>"
  echo "Provide the instance number to start (0 to $((MAX_INSTANCES - 1)))."
  exit 1
fi

INSTANCE_NUMBER=$1

# Validate the instance number
if (( INSTANCE_NUMBER < 0 || INSTANCE_NUMBER >= MAX_INSTANCES )); then
  echo "Error: Instance number must be between 0 and $((MAX_INSTANCES - 1))."
  exit 1
fi

# Compute CPU and GPU offsets for the selected instance
CPU_START=$(( INSTANCE_NUMBER * CPUS_PER_INSTANCE ))
CPU_END=$(( CPU_START + CPUS_PER_INSTANCE - 1 ))
CPU_SET="${CPU_START}-${CPU_END}"

# Validate resource availability
if (( CPU_START + CPUS_PER_INSTANCE > TOTAL_CPUS )); then
  echo "Error: Not enough CPUs available for instance $INSTANCE_NUMBER."
  exit 1
fi

if (( INSTANCE_NUMBER >= TOTAL_GPUS )); then
  echo "Error: Not enough GPUs available for instance $INSTANCE_NUMBER."
  exit 1
fi

# Start the specific instance
echo "Starting instance $INSTANCE_NUMBER with CPU_SET=${CPU_SET} and GPU=${INSTANCE_NUMBER}"

# Run docker-compose for the specific instance
make run INSTANCE_ID=$INSTANCE_NUMBER CPU_SET=$CPU_SET

sleep 10

echo "Instance $INSTANCE_NUMBER started successfully."

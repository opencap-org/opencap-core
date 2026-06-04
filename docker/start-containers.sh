#!/bin/bash

# Configuration
MAX_INSTANCES=8
CPUS_PER_INSTANCE=14
GPUS_PER_INSTANCE=1

# Get the total number of CPUs and GPUs available
TOTAL_CPUS=$(nproc)
TOTAL_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Read number of instances to start
if [ -z "$1" ]; then
  echo "Usage: $0 <number_of_instances>"
  echo "Provide the number of instances to start (max $MAX_INSTANCES)."
  exit 1
fi

NUM_INSTANCES=$1

# Validate the number of instances
if (( NUM_INSTANCES > MAX_INSTANCES )); then
  echo "Error: Maximum number of instances is $MAX_INSTANCES."
  exit 1
fi

# Check if there are enough resources
if (( NUM_INSTANCES * CPUS_PER_INSTANCE > TOTAL_CPUS )); then
  echo "Error: Not enough CPUs. Required: $((NUM_INSTANCES * CPUS_PER_INSTANCE)), Available: $TOTAL_CPUS."
  exit 1
fi

if (( NUM_INSTANCES * GPUS_PER_INSTANCE > TOTAL_GPUS )); then
  echo "Error: Not enough GPUs. Required: $((NUM_INSTANCES * GPUS_PER_INSTANCE)), Available: $TOTAL_GPUS."
  exit 1
fi

# Display summary
echo "Starting $NUM_INSTANCES instances..."
echo "Total CPUs: $TOTAL_CPUS (using $CPUS_PER_INSTANCE per instance)"
echo "Total GPUs: $TOTAL_GPUS (using $GPUS_PER_INSTANCE per instance)"
echo

# Start instances
for (( i=0; i<NUM_INSTANCES; i++ )); do
  INSTANCE_ID=$i
  CPU_START=$(( i * CPUS_PER_INSTANCE ))
  CPU_END=$(( CPU_START + CPUS_PER_INSTANCE - 1 ))
  CPU_SET="${CPU_START}-${CPU_END}"

  echo "Starting instance $INSTANCE_ID with CPU_SET=${CPU_SET} and GPU=${INSTANCE_ID}"

  # Run docker-compose for each instance
  make run INSTANCE_ID=$INSTANCE_ID CPU_SET=$CPU_SET

  sleep 2
done

echo "All instances started successfully."


#!/bin/bash

# Function to check if a container is running
is_container_alive() {
  local container_name=$1
  docker ps --filter "name=^/${container_name}$" --filter "status=running" --format '{{.Names}}' | grep -wq "$container_name"
  return $?
}

# Loop through numbers 0 to 7
for n in {0..7}; do
  # Container names
  opencap_openpose="opencap_${n}-openpose-1"
  opencap_mmpose="opencap_${n}-mmpose-1"
  opencap_mobilecap="opencap_${n}-mobilecap-1"

  # Check if all three containers are alive
  if is_container_alive "$opencap_openpose" && \
     is_container_alive "$opencap_mmpose" && \
     is_container_alive "$opencap_mobilecap"; then
    echo "All containers for instance $n are alive. Skipping."
    continue
  fi

  # Check if any container exists
  if docker ps -a --filter "name=^/opencap_${n}-(openpose|mmpose|mobilecap)-1$" --format '{{.Names}}' | grep -q "opencap_${n}"; then
    echo "Some containers for instance $n are not alive. Stopping instance."
    ./stop-container.sh "$n"
    ./start-container.sh "$n"
  else
    echo "No containers for instance $n. Skipping."
  fi

done

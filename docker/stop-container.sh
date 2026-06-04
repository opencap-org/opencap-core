#!/bin/bash

# Check if INSTANCE_ID is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <INSTANCE_ID>"
  exit 1
fi

INSTANCE_ID=$1
COMPOSE_PROJECT_NAME="opencap_${INSTANCE_ID}"

echo "Stopping and removing containers for INSTANCE_ID=${INSTANCE_ID}..."

# Stop and remove containers associated with the project
docker-compose \
  --project-name $COMPOSE_PROJECT_NAME \
  down

# Verify if containers are removed
if [ $? -eq 0 ]; then
  echo "Successfully stopped and removed containers for INSTANCE_ID=${INSTANCE_ID}."
else
  echo "Failed to stop and remove containers for INSTANCE_ID=${INSTANCE_ID}."
fi


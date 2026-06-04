BASE_NAME := 660440363484.dkr.ecr.us-west-2.amazonaws.com
REPO_NAME := opencap
PROD_BRANCH := main
DEV_BRANCH := dev

# Initialize variables if not passed in
INSTANCE_ID ?= 0
CPU_SET ?= ""

# Determine the branch name
CURRENT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

# Determine image tag and name based on branch
# If not 'main' or 'dev', then build local image
ifeq ($(CURRENT_BRANCH),$(PROD_BRANCH))
	OPENCAP_IMAGE_TAG := opencap
	OPENPOSE_IMAGE_TAG := openpose
	MMPOSE_IMAGE_TAG := mmpose

	OPENCAP_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(OPENCAP_IMAGE_TAG)
	OPENPOSE_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(OPENPOSE_IMAGE_TAG)
	MMPOSE_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(MMPOSE_IMAGE_TAG)

else ifeq ($(CURRENT_BRANCH),$(DEV_BRANCH))
	OPENCAP_IMAGE_TAG := opencap-dev
	OPENPOSE_IMAGE_TAG := openpose-dev
	MMPOSE_IMAGE_TAG := mmpose-dev

	OPENCAP_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(OPENCAP_IMAGE_TAG)
	OPENPOSE_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(OPENPOSE_IMAGE_TAG)
	MMPOSE_IMAGE_NAME := $(BASE_NAME)/$(REPO_NAME)/$(MMPOSE_IMAGE_TAG)

else
	OPENCAP_IMAGE_NAME := opencap-local
	OPENPOSE_IMAGE_NAME := openpose-local
	MMPOSE_IMAGE_NAME := mmpose-local

endif

# Get git commit hash info to pass it into container
GIT_COMMIT_HASH=$(shell git rev-parse --short HEAD)

.PHONY: build
build:
	wget -c -O ../mmpose/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
	wget -c -O ../mmpose/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth https://mc-opencap-public.s3.us-west-2.amazonaws.com/mmpose_pth/hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth
	
	docker build --build-arg GIT_COMMIT_HASH=$(GIT_COMMIT_HASH) -t $(OPENCAP_IMAGE_NAME) .. -f Dockerfile
	docker build -t $(OPENPOSE_IMAGE_NAME) .. -f openpose/Dockerfile
	docker build -t $(MMPOSE_IMAGE_NAME) .. -f mmpose/Dockerfile

.PHONY: push
push:
ifeq ($(CURRENT_BRANCH),$(PROD_BRANCH))
	aws ecr get-login-password --region us-west-2 --profile opencap | docker login --username AWS --password-stdin 660440363484.dkr.ecr.us-west-2.amazonaws.com

	docker push $(OPENCAP_IMAGE_NAME)
	docker push $(OPENPOSE_IMAGE_NAME)
	docker push $(MMPOSE_IMAGE_NAME)

else ifeq ($(CURRENT_BRANCH),$(DEV_BRANCH))
	aws ecr get-login-password --region us-west-2 --profile opencap | docker login --username AWS --password-stdin 660440363484.dkr.ecr.us-west-2.amazonaws.com

	docker push $(OPENCAP_IMAGE_NAME)
	docker push $(OPENPOSE_IMAGE_NAME)
	docker push $(MMPOSE_IMAGE_NAME)

else
	@echo "Git branch is not 'main' or 'dev', skipping push step"

endif

.PHONY: run
run:
	@echo "Usage: sudo make run INSTANCE_ID=<unique_id> CPU_SET=<cpu_set>"
	@echo "Defaults: INSTANCE_ID=0, CPU_SET=\"\""	

	COMPOSE_PROJECT_NAME=opencap_$(INSTANCE_ID) \
	OPENCAP_IMAGE_NAME=$(OPENCAP_IMAGE_NAME) \
	OPENPOSE_IMAGE_NAME=$(OPENPOSE_IMAGE_NAME) \
	MMPOSE_IMAGE_NAME=$(MMPOSE_IMAGE_NAME) \
	INSTANCE_ID=$(INSTANCE_ID) \
	CPU_SET=$(CPU_SET) \
	docker compose up -d

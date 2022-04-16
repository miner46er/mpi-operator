DOCKERCMD=docker
BUILD_VERSION?=v0.1.12-autoscale
IMAGE_REGISTRY?=localhost:32000
IMAGE_BASE_NAME?=horovod
IMAGE_COMMON_BUILD_ARGS?= \
	--build-arg version=$(BUILD_VERSION) \
	--build-arg http_proxy=$(http_proxy) \
	--build-arg https_proxy=$(https_proxy) \
	--build-arg no_proxy=$(no_proxy)

.PHONY: image image_gpu image_cpu

image: image_gpu image_cpu

image_gpu:
	$(DOCKERCMD) build $(IMAGE_COMMON_BUILD_ARGS) -t $(IMAGE_REGISTRY)/$(IMAGE_BASE_NAME):$(BUILD_VERSION) . -f elastic.Dockerfile

image_cpu:
	$(DOCKERCMD) build $(IMAGE_COMMON_BUILD_ARGS) -t $(IMAGE_REGISTRY)/$(IMAGE_BASE_NAME):$(BUILD_VERSION)-cpu . -f elastic-cpu.Dockerfile

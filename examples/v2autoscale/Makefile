DOCKERCMD=docker
BUILD_VERSION?=v0.1.46-autoscale
IMAGE_REGISTRY?=us-central1-docker.pkg.dev/ml-autoscaling/ml-autoscaling-image-registry
#IMAGE_REGISTRY?=localhost:32000
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

image_push: image_push_gpu image_push_cpu

image_push_gpu:
	$(DOCKERCMD) push $(IMAGE_REGISTRY)/$(IMAGE_BASE_NAME):$(BUILD_VERSION)

image_push_cpu:
	$(DOCKERCMD) push $(IMAGE_REGISTRY)/$(IMAGE_BASE_NAME):$(BUILD_VERSION)-cpu

image_cpu_dev: image_cpu image_push_cpu

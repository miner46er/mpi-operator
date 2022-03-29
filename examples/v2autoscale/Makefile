DOCKERCMD=docker
BUILD_VERSION?=v0.1.1-autoscale
IMAGE_COMMON_BUILD_ARGS?=" \
	--build-arg version=$(CONTROLLER_VERSION) \
	--build-arg http_proxy=$(http_proxy) \
	--build-arg https_proxy=$(https_proxy) \
	--build-arg no_proxy=$(no_proxy)"

.PHONY: image
image:
	$(DOCKERCMD) build $(IMAGE_COMMON_BUILD_ARGS) -t localhost:32000/horovod:$(BUILD_VERSION) .

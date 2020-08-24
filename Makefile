APP_NAME=amirassov/gwd
CONTAINER_NAME=gwd
PROJECT_NAME=/global-wheat-detection

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

build:  ## Build the container
	nvidia-docker build -t ${APP_NAME}$(MODE) -f Dockerfile$(MODE) .

run-extreme: ## Run container in extreme
	nvidia-docker run \
		-e DISPLAY=unix${DISPLAY} -v /tmp/.X11-unix:/tmp/.X11-unix --privileged \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME}$(MODE) \
		-v $(shell pwd):${PROJECT_NAME} \
		-v /home/amirassov/global_wheat_data:/data_old \
		-v /home/amirassov/global_wheat_data_test:/data \
		-v /home/amirassov/global_wheat_dumps:/dumps ${APP_NAME}$(MODE) bash

run-dgx: ## Run container in dgx
	nvidia-docker run \
		--ipc=host \
		-itd \
		--name=${CONTAINER_NAME}$(MODE) \
		-v $(shell pwd):${PROJECT_NAME} \
		-v /raid/data_share/amirassov/global_wheat_data:/data \
		-v /raid/data_share/amirassov/global_wheat_dumps:/dumps ${APP_NAME}$(MODE) bash

exec: ## Run a bash in a running container
	nvidia-docker exec -it ${CONTAINER_NAME}$(MODE) bash

stop: ## Stop and remove a running container
	docker stop ${CONTAINER_NAME}$(MODE); docker rm ${CONTAINER_NAME}$(MODE)

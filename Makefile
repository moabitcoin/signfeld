dockerimage ?= das/synthetic-signs
dockerfile ?= Dockerfile
srcdir ?= $(shell pwd)
datadir ?= $(shell pwd)

docker-install:
	@docker build -t $(dockerimage) -f $(dockerfile) .

di: docker-install

docker-update:
	@docker build -t $(dockerimage) -f $(dockerfile) . --pull --no-cache

du: docker-update

docker-run:
	@docker run                              \
	  --runtime=nvidia                       \
	  --ipc=host                             \
	  -it                                    \
	  --rm                                   \
	  -v $(srcdir):/usr/src/app/             \
	  -v $(datadir):/data                    \
	  $(dockerimage)

dr: docker-run

.PHONY: docker-install di docker-run dr docker-update du

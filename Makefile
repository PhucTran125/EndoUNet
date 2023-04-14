build-proto: proto/detect.proto
	python3 -m grpc_tools.protoc -I=proto\
		--python_out=polypnet/grpc\
		--grpc_python_out=polypnet/grpc\
		proto/*.proto

build:
	docker-compose build

pull:
	docker-compose pull

push:
	docker-compose push

stop:
	docker-compose down

dev: build
	docker-compose -f docker-compose.dev.yml down
	docker-compose -f docker-compose.dev.yml up -d
	docker-compose -f docker-compose.dev.yml logs -f || true

deploy: stop
	docker-compose -f docker-compose.yml up -d

test: build
	docker-compose -f docker-compose.yml\
		-f docker-compose.dev.yml\
		run --rm worker pytest --cov=. tests/

pull-models:
	gsutil rsync -R gs://vdsense-model-repo/polypnet-models/polypnet ./models

push-models:
	gsutil rsync -R ./models gs://vdsense-model-repo/polypnet-models/polypnet

RFLAGS=
# RTARGET=s@172.18.0.1
RTARGET=sangdv@202.191.56.249

RTARGET_DIR=~/polyp-service/

rpush:
	rsync -avh --progress $(RFLAGS) \
		--exclude-from='.pushignore' \
		. $(RTARGET):$(RTARGET_DIR)

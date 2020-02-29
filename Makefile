build:
	docker-compose build

pull:
	docker-compose pull

push:
	docker-compose push

stop:
	docker-compose down

dev: build
	docker-compose -f docker-compose.yml\
		-f docker-compose.dev.yml down
	docker-compose -f docker-compose.yml\
		-f docker-compose.dev.yml up -d
	docker-compose logs -f || true

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

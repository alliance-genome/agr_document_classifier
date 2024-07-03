ifndef ENV_FILE
	ENV_FILE=.env
endif

include ${ENV_FILE}

ifndef CLASSIFIER_MODEL_NAME
	CLASSIFIER_MODEL_NAME="classifier"
endif

train:
	docker-compose --env-file ${ENV_FILE} run agr_document_classifier python agr_document_classifier.py --mode train --embedding_model_path /usr/src/app/embedding_model --training_docs_dir /usr/src/app/training --test_size $(TEST_SIZE) --classifier_model_path /usr/src/app/classifiers/$(CLASSIFIER_MODEL_NAME)

classify:
	docker-compose --env-file ${ENV_FILE} run agr_document_classifier python agr_document_classifier.py --mode classify --embedding_model_path /usr/src/app/embedding_model --classifier_model_path /usr/src/app/classifiers/$(CLASSIFIER_MODEL_NAME) --classify_docs_dir /usr/src/app/classification

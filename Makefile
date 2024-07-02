ifndef ENV_FILE
	ENV_FILE=.env
endif

include ${ENV_FILE}

ifndef CLASSIFIER_MODEL_NAME
	CLASSIFIER_MODEL_NAME="classifier"
endif

train:
	docker-compose --env-file ${ENV_FILE} run agr_document_classifier python agr_document_classifier.py --mode train --embedding_model_path $(EMBEDDING_MODEL_PATH) --training_docs_dir $(TRAINING_DOCS_DIR) --test_size $(TEST_SIZE) --classifier_model_path $(CLASSIFIER_MODEL_PATH)/$(CLASSIFIER_MODEL_NAME)

classify:
	docker-compose --env-file ${ENV_FILE} run agr_document_classifier python agr_document_classifier.py --mode classify --embedding_model_path $(EMBEDDING_MODEL_PATH) --classifier_model_path $(CLASSIFIER_MODEL_PATH)/$(CLASSIFIER_MODEL_NAME) --classify_docs_dir $(CLASSIFY_DOCS_DIR)

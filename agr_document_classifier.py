import argparse
import glob
import logging
import os
import os.path
import re
from pathlib import Path

import fasttext
import numpy as np
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File
from joblib import dump, load
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


def extract_embeddings(model, document):
    # Split the document into words and extract the embedding for each word
    words = document.split()
    embeddings = [model.get_word_vector(word) for word in words]
    # Average the embeddings to get a single vector for the document
    doc_embedding = np.mean(embeddings, axis=0)
    return doc_embedding


def load_embedding_model(model_path):
    logger.info("Loading embeddings...")
    model = fasttext.load_model(model_path)
    logger.info("Finished loading embeddings.")
    return model


def train_classifier(embedding_model_path: str, training_data_dir: str, test_size: float):
    model = load_embedding_model(model_path=embedding_model_path)
    classifier = SVC(random_state=42)

    X = []
    y = []

    # Assume you have a function to get your training data
    # For each document in your training data, extract embeddings and labels
    for label in ["positive", "negative"]:
        for document in get_documents(os.path.join(training_data_dir, label, "*")):
            if document:
                doc_embedding = extract_embeddings(model, document)
                X.append(doc_embedding)
                y.append(int(label == "positive"))

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Define the parameter grid to search
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto'],
        'degree': [3, 4, 5]  # Only used for 'poly' kernel
    }

    # Define the parameter grid to search

    # Initialize the grid search
    grid_search = GridSearchCV(estimator=classifier, param_grid=param_grid, cv=5, scoring='f1', verbose=1, n_jobs=-1)

    # Perform the grid search on the data
    grid_search.fit(X, y)

    # Get the best parameters and the best score
    best_params = grid_search.best_params_

    # Train the final model on the entire dataset using the best parameters
    best_classifier = SVC(**best_params, random_state=42).fit(X, y)

    # Define the number of folds for cross-validation
    k_folds = 5

    # Perform cross-validation with multiple scoring metrics
    scoring_metrics = ['precision', 'recall', 'f1']
    cross_val_results = cross_validate(best_classifier, X, y, cv=k_folds, scoring=scoring_metrics)

    # Calculate the average scores across all folds
    average_precision = cross_val_results['test_precision'].mean()
    average_recall = cross_val_results['test_recall'].mean()
    average_fscore = cross_val_results['test_f1'].mean()

    # Train the final model on the entire dataset
    final_classifier = classifier.fit(X, y)

    # Return the trained model and performance metrics
    return final_classifier, average_precision, average_recall, average_fscore


def save_classifier(classifier, file_path):
    dump(classifier, file_path)


def load_classifier(file_path):
    return load(file_path)


def get_documents(input_docs_dir: str):
    client = None
    for file_path in glob.glob(input_docs_dir):
        file_obj = Path(file_path)
        if file_path.endswith(".tei.xml") or file_path.endswith(".pdf"):
            with file_obj.open("rb") as fin:
                try:
                    if file_path.endswith(".pdf"):
                        if client is None:
                            client = Client(base_url=os.environ.get("GROBID_API_URL"), timeout=1000, verify_ssl=False)
                        logger.info("Started pdf to TEI conversion")
                        form = ProcessForm(
                            segment_sentences="1",
                            input_=File(file_name=file_obj.name, payload=fin, mime_type="application/pdf"))
                        r = process_fulltext_document.sync_detailed(client=client, multipart_data=form)
                        file_stream = r.content
                    else:
                        file_stream = fin
                    article: Article = TEI.parse(file_stream, figures=True)
                    sentences = [re.sub('<[^<]+>', '', sentence.text) for section in article.sections for paragraph
                                 in section.paragraphs for sentence in paragraph if not sentence.text.isdigit() and
                                 not (
                                         len(section.paragraphs) == 3 and
                                         section.paragraphs[0][0].text in ['\n', ' '] and
                                         section.paragraphs[-1][0].text in ['\n', ' ']
                                 )]
                    sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
                except Exception as e:
                    logging.error(str(e))
                    continue
                yield " ".join(sentences)


def classify_documents(embedding_model_path: str, classifier_model_path: str, input_docs_dir: str):
    embedding_model = load_embedding_model(model_path=embedding_model_path)
    classifier_model = load_classifier(classifier_model_path)
    X = []
    for document in get_documents(input_docs_dir=input_docs_dir):
        doc_embedding = extract_embeddings(embedding_model, document)
        X.append(doc_embedding)

    X = np.array(X)
    return classifier_model.predict(X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify documents or train document classifiers')
    parser.add_argument("-m", "--mode", type=str, choices=['train', 'classify'], default="classify",
                        help="Mode of operation: train or classify")
    parser.add_argument("-e", "--embedding_model_path", type=str, help="Path to the word embedding model")
    parser.add_argument("-c", "--classifier_model_path", type=str, help="Path to the classifier model")
    parser.add_argument("-i", "--classify_docs_dir", type=str, help="Path to the docs to classify",
                        required=False)
    parser.add_argument("-t", "--training_docs_dir", type=str, help="Path to the docs to classify",
                        required=False)
    parser.add_argument("-s", "--test_size", type=float, help="Percentage of data used for testing",
                        default=0.2)

    args = parser.parse_args()
    if args.mode == "classify":
        classifications = classify_documents(embedding_model_path=args.embedding_model_path,
                                             classifier_model_path=args.classifier_model_path,
                                             input_docs_dir=args.classify_docs_dir)
        print(classifications)
    else:
        classifier, precision, recall, fscore = train_classifier(embedding_model_path=args.embedding_model_path,
                                                                 training_data_dir=args.training_docs_dir,
                                                                 test_size=args.test_size)
        save_classifier(classifier=classifier, file_path=args.classifier_model_path)
        print(f"Precision: {str(precision)}, Recall: {str(recall)}, F1 score: {str(fscore)}")


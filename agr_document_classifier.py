import argparse
import glob
import logging
import os
import os.path
import re
from pathlib import Path
from typing import Tuple

import fasttext
import nltk
import numpy as np
from grobid_client import Client
from grobid_client.api.pdf import process_fulltext_document
from grobid_client.models import Article, ProcessForm
from grobid_client.types import TEI, File
from joblib import dump, load
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


nltk.download('stopwords')
nltk.download('punkt')


logger = logging.getLogger(__name__)


POSSIBLE_CLASSIFIERS = {
    'LogisticRegression': {
        'model': LogisticRegression(random_state=42),
        'params': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga']
        }
    },
    'RandomForestClassifier': {
        'model': RandomForestClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    },
    'GradientBoostingClassifier': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7]
        }
    },
    'XGBClassifier': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'max_depth': [3, 4, 5, 6, 7]
        }
    },
    'MLPClassifier': {
        'model': MLPClassifier(max_iter=1000),
        'params': {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'learning_rate_init': [0.001, 0.01, 0.1]
        }
    },
    'SVC': {
        'model': SVC(),
        'params': {
            'C': [0.1, 1, 10, 100, 1000],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['scale', 'auto', 0.001, 0.0001],
            'degree': [2, 3, 4, 5],  # Only used if kernel is 'poly'
            'coef0': [0.0, 0.1, 0.5, 1.0]  # Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
        }
    }
}


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
    embedding_model = load_embedding_model(model_path=embedding_model_path)

    X = []
    y = []

    # Assume you have a function to get your training data
    # For each document in your training data, extract embeddings and labels
    logger.info("Loading training set.")
    for label in ["positive", "negative"]:
        for fulltext, title, abstract in get_documents(os.path.join(training_data_dir, label, "*")):
            text = abstract
            if text:
                text = remove_stopwords(text)
                text_embedding = extract_embeddings(embedding_model, text)
                X.append(text_embedding)
                y.append(int(label == "positive"))
    logger.info("Finished loading training set.")
    logger.info(f"Dataset size: {str(len(X))}")

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    best_score = 0
    best_classifier = None
    best_params = None
    best_classifier_name = ""

    logger.info("Starting model selection with grid search hyperparameter optimization and cross-validation.")
    # Iterate over classifiers and perform grid search
    for classifier_name, classifier_info in POSSIBLE_CLASSIFIERS.items():
        logger.info(f"Evaluating model {classifier_name}.")
        grid_search = GridSearchCV(estimator=classifier_info['model'], param_grid=classifier_info['params'], cv=5,
                                   scoring='f1', verbose=1, n_jobs=-1)
        grid_search.fit(X, y)

        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_classifier = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_classifier_name = classifier_name

    logger.info(f"Selected model {best_classifier_name}.")
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
    final_classifier = best_classifier.fit(X, y)

    # Return the trained model and performance metrics
    return final_classifier, average_precision, average_recall, average_fscore, best_classifier_name, best_params


def save_classifier(classifier, file_path):
    dump(classifier, file_path)


def load_classifier(file_path):
    return load(file_path)


def get_sentences_from_tei_section(section):
    sentences = []
    for paragraph in section.paragraphs:
        for sentence in paragraph:
            if not sentence.text.isdigit() and not (
                    len(section.paragraphs) == 3 and
                    section.paragraphs[0][0].text in ['\n', ' '] and
                    section.paragraphs[-1][0].text in ['\n', ' ']
            ):
                sentences.append(re.sub('<[^<]+>', '', sentence.text))
    sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
    return sentences


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def get_documents(input_docs_dir: str) -> Tuple[str, str, str]:
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
                    sentences = []
                    for section in article.sections:
                        sentences.extend(get_sentences_from_tei_section(section))
                    abstract = ""
                    for section in article.sections:
                        if section.name == "ABSTRACT":
                            abstract = " ".join(get_sentences_from_tei_section(section))
                            break
                    yield " ".join(sentences), article.title, abstract
                except Exception as e:
                    logging.error(str(e))
                    continue


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
    parser.add_argument("-l", "--log_level", type=str,
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help="Set the logging level")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), 0))

    if args.mode == "classify":
        classifications = classify_documents(embedding_model_path=args.embedding_model_path,
                                             classifier_model_path=args.classifier_model_path,
                                             input_docs_dir=args.classify_docs_dir)
        print(classifications)
    else:
        classifier, precision, recall, fscore, classifier_name, classifier_params = train_classifier(
            embedding_model_path=args.embedding_model_path, training_data_dir=args.training_docs_dir,
            test_size=args.test_size)
        save_classifier(classifier=classifier, file_path=args.classifier_model_path)
        print(f"Selected Model: {classifier_name}, Parameters {classifier_params}, "
              f"Precision: {str(precision)}, Recall: {str(recall)}, F1 score: {str(fscore)}")


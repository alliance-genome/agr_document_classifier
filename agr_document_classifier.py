import argparse
import glob
import os
import os.path
import re

import fasttext
import numpy as np
from grobid_client.models import Article
from grobid_client.types import TEI
from joblib import dump, load
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def extract_embeddings(model, document):
    # Split the document into words and extract the embedding for each word
    words = document.split()
    embeddings = [model.get_word_vector(word) for word in words]
    # Average the embeddings to get a single vector for the document
    doc_embedding = np.mean(embeddings, axis=0)
    return doc_embedding


def train_classifier(embedding_model_path: str, training_data_dir: str, test_size: float):
    model = fasttext.load_model(embedding_model_path)
    classifier = MLPClassifier(hidden_layer_sizes=(512,), max_iter=500)

    X = []
    y = []

    # Assume you have a function to get your training data
    # For each document in your training data, extract embeddings and labels
    for label in ["positive", "negative"]:
        for document in get_documents(os.path.join(training_data_dir, label)):
            doc_embedding = extract_embeddings(model, document)
            X.append(doc_embedding)
            y.append(label)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Predict on the test set
    y_pred = classifier.predict(X_test)

    # Calculate precision, recall, and F1-score
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

    # Return the trained model and performance metrics
    return classifier, precision, recall, fscore


def save_classifier(classifier, file_path):
    dump(classifier, file_path)


def load_classifier(file_path):
    return load(file_path)


def get_documents(input_docs_dir: str):
    for file_path in glob.glob(input_docs_dir):
        with open(file_path, "rb") as fin:
            article: Article = TEI.parse(fin, figures=True)
            sentences = [re.sub('<[^<]+>', '', sentence.text) for section in article.sections for paragraph
                         in section.paragraphs for sentence in paragraph if not sentence.text.isdigit() and
                         not (
                                 len(section.paragraphs) == 3 and
                                 section.paragraphs[0][0].text in ['\n', ' '] and
                                 section.paragraphs[-1][0].text in ['\n', ' ']
                         )]
            sentences = [sentence if sentence.endswith(".") else f"{sentence}." for sentence in sentences]
            return " ".join(sentences)


def classify_documents(embedding_model_path: str, classifier_model_path: str, input_docs_dir: str):
    embedding_model = fasttext.load_model(embedding_model_path)
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
    parser.add_argument("-i", "--input_docs_dir", type=str, help="Path to the input docs directory")
    parser.add_argument("-t", "--test_size", type=float, help="Percentage of data used for testing",
                        default=0.2)

    args = parser.parse_args()
    if args.mode == "classify":
        classifications = classify_documents(embedding_model_path=args.embedding_model_path,
                                             classifier_model_path=args.classifier_model_path,
                                             input_docs_dir=args.input_docs_dir)
        print(classifications)
    else:
        classifier, precision, recall, fscore = train_classifier(embedding_model_path=args.embedding_model_path,
                                                                 training_data_dir=args.input_docs_dir,
                                                                 test_size=0.4)
        save_classifier(classifier=classifier, file_path=args.classifier_model_path)


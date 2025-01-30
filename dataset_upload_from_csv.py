import csv
import json
import logging
import os
import requests
from typing import List

from abc_utils import get_curie_from_xref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
blue_api_base_url = os.environ.get('API_SERVER', "literature-rest.alliancegenome.org")
CREATE_DATASET_URL = f"https://{blue_api_base_url}/datasets/"
ADD_ENTRY_URL = f"https://{blue_api_base_url}/datasets/data_entry/"


# Function to create a dataset
def create_dataset(title: str, description: str, mod_abbreviation: str, topic: str, task_type: str) -> (int, int):
    payload = {
        "title": title,
        "description": description,
        "mod_abbreviation": mod_abbreviation,
        "data_type": topic,
        "dataset_type": task_type
    }
    response = requests.post(CREATE_DATASET_URL, json=payload)
    if response.status_code == 201:
        dataset_id = response.json()["id"]
        version = response.json()["version"]
        logger.info(f"Dataset created with ID: {dataset_id}")
        return dataset_id, version
    else:
        logger.error(f"Failed to create dataset: {response.text}")
        response.raise_for_status()


# Function to add an entry to the dataset
def add_entry_to_dataset(mod_abbreviation: str, topic: str, task_type: str, version: int, reference_curie: str,
                         positive: bool):
    payload = {
        "mod_abbreviation": mod_abbreviation,
        "data_type": topic,
        "dataset_type": task_type,
        "version": version,
        "reference_curie": reference_curie,
        "positive": positive
    }
    response = requests.post(ADD_ENTRY_URL, json=payload)
    if response.status_code == 201:
        logger.info("Entry added to dataset")
    else:
        logger.error(f"Failed to add entry to dataset: {response.text}")
        response.raise_for_status()


# Function to upload dataset from CSV
def upload_dataset_from_csv(csv_file: str, title: str, description: str, mod_abbreviation: str, topic: str):
    dataset_id, version = create_dataset(title=title, description=description, mod_abbreviation=mod_abbreviation,
                                         topic=topic, task_type="document_classification")
    with open(csv_file, 'r') as file:
        csv_reader = csv.DictReader(file, delimiter=',')  # Change delimiter to comma
        for row in csv_reader:
            agrkb_id = row.get('AGRKBID')
            positive = bool(row.get('Positive/Negative'))
            if not agrkb_id:
                xref = row.get('XREF')
                agrkb_id = get_curie_from_xref(xref)
                if not agrkb_id or not positive:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue
            add_entry_to_dataset(mod_abbreviation=mod_abbreviation, topic=topic, task_type="document_classification",
                                 version=version, reference_curie=agrkb_id, positive=positive)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload dataset to ABC from CSV file")
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument("title", help="Name of the dataset")
    parser.add_argument("description", help="Description of the dataset")
    parser.add_argument("mod_abbreviation", help="Mod abbreviation")
    parser.add_argument("topic", help="Topic of the dataset")
    args = parser.parse_args()

    upload_dataset_from_csv(args.csv_file, args.title, args.description, args.mod_abbreviation, args.topic)

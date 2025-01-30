import csv
import logging

from abc_utils import get_curie_from_xref, create_dataset, add_entry_to_dataset

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Function to upload dataset from CSV
def upload_dataset_from_csv(csv_file: str, title: str, description: str, mod_abbreviation: str, topic: str):
    dataset_id, version = create_dataset(title=title, description=description, mod_abbreviation=mod_abbreviation,
                                         topic=topic, dataset_type="document")
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

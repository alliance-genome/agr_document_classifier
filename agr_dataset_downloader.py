import argparse
import csv
import json
import logging
import os
import requests
import urllib.request
from urllib.error import HTTPError

from fastapi_okta.okta_utils import get_authentication_token, generate_headers

from abc_utils import get_file_from_abc_reffile_obj, get_curie_from_xref

logger = logging.getLogger(__name__)

blue_api_base_url = os.environ.get('API_SERVER', "literature-rest.alliancegenome.org")


def download_pdf_files(agr_curie, file_name, output_dir):
    all_reffiles_for_pap_api = f'https://{blue_api_base_url}/reference/referencefile/show_all/{agr_curie}'
    request = urllib.request.Request(url=all_reffiles_for_pap_api)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            for ref_file in resp_obj:
                if ref_file["file_extension"] == "pdf" and ref_file["file_class"] == "main":
                    file_content = get_file_from_abc_reffile_obj(ref_file)
                    with open(os.path.join(output_dir, file_name + ".pdf"), "wb") as out_file:
                        out_file.write(file_content)
    except HTTPError as e:
        logger.error(e)


def download_and_categorize_pdfs(csv_file, output_dir):
    os.makedirs(os.path.join(output_dir, "positive"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "negative"), exist_ok=True)

    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        for row in csv_reader:
            if len(row) < 3:
                logger.warning(f"Skipping invalid row: {row}")
                continue

            wbpaper_id, label = row[1], row[2]
            agr_curie = get_curie_from_xref(wbpaper_id)

            if agr_curie:
                category = "positive" if label.lower() == "positive" else "negative"
                file_name = wbpaper_id.replace(":", "_")
                category_dir = os.path.join(output_dir, category)
                download_pdf_files(agr_curie, file_name, category_dir)
            else:
                logger.warning(f"Could not find AGR curie for {wbpaper_id}")


def main():
    parser = argparse.ArgumentParser(description="Download and categorize PDFs from a CSV file")
    parser.add_argument("-f", "--csv-file", required=True, help="Path to the input CSV file")
    parser.add_argument("-o", "--output-dir", default="downloaded_files", help="Output directory for downloaded files")
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    download_and_categorize_pdfs(args.csv_file, out_dir)


if __name__ == '__main__':
    main()

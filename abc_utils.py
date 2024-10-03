import json
import logging
import os
import urllib.request
import requests
from typing import List
from urllib.error import HTTPError

from cachetools import TTLCache
from fastapi_okta.okta_utils import get_authentication_token, generate_headers
from tqdm import tqdm

blue_api_base_url = os.environ.get('API_SERVER', "literature-rest.alliancegenome.org")


logger = logging.getLogger(__name__)


cache = TTLCache(maxsize=100, ttl=7200)


job_category_topic_map = {
    "catalytic_activity": "ATP:0000061"
}


def get_mod_species_map():
    url = f'https://{blue_api_base_url}/mod/taxons/default'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return {mod["mod_abbreviation"]: mod["taxon_id"] for mod in resp_obj}
    except HTTPError as e:
        logger.error(e)


def get_mod_id_from_abbreviation(mod_abbreviation):
    url = f'https://{blue_api_base_url}/mod/{mod_abbreviation}'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj["mod_id"]
    except HTTPError as e:
        logger.error(e)


def get_cached_mod_species_map():
    if 'mod_species_map' not in cache:
        cache['mod_species_map'] = get_mod_species_map()
    return cache['mod_species_map']


def get_cached_mod_id_from_abbreviation(mod_abbreviation):
    if 'mod_abbreviation_id' not in cache:
        cache['mod_abbreviation_id'] = {}
    if mod_abbreviation not in cache['mod_abbreviation_id']:
        cache['mod_abbreviation_id'][mod_abbreviation] = get_mod_id_from_abbreviation(mod_abbreviation)
    return cache['mod_abbreviation_id'][mod_abbreviation]


def get_cached_mod_abbreviation_from_id(mod_id):
    if 'mod_id_abbreviation' not in cache:
        cache['mod_id_abbreviation'] = {}
        for mod_abbreviation in get_cached_mod_species_map().keys():
            cache['mod_id_abbreviation'][get_cached_mod_id_from_abbreviation(mod_abbreviation)] = mod_abbreviation
    return cache['mod_id_abbreviation'][mod_id]


def get_curie_from_reference_id(reference_id):
    url = f'https://{blue_api_base_url}/reference/{reference_id}'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj["curie"]
    except HTTPError as e:
        logger.error(e)


def send_classification_tag_to_abc(reference_curie: str, mod_abbreviation: str, topic: str, negated: bool,
                                   confidence_level: str):
    url = f'https://{blue_api_base_url}/topic_entity_tag'
    token = get_authentication_token()
    tet_data = {
        "created_by": "default_user",
        "updated_by": "default_user",
        "topic": topic,
        "species": get_cached_mod_species_map()[mod_abbreviation],
        "topic_entity_tag_source_id": "TO_DEFINE",
        "negated": negated,
        "confidence_level": confidence_level,
        "reference_curie": reference_curie,
        "force_insertion": True
    }
    headers = generate_headers(token)
    try:
        response = requests.request("POST", url=url, json=tet_data, headers=headers)
        if response.status_code == 200:
            logger.info("TET created")
        else:
            logger.error(f"Failed to create TET: {str(tet_data)}")
    except requests.exceptions.RequestException as e:
        logger.info(f"Error occurred during TET upload: {e}")
        return None


def get_training_set_from_abc(mod_abbreviation: str, topic: str):
    ...


def get_jobs_to_classify(limit: int = 1000, offset: int = 0):
    jobs_url = f'https://{blue_api_base_url}/workflow_tag/jobs/classification_job?limit={limit}&offset={offset}'
    request = urllib.request.Request(url=jobs_url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(request) as response:
            resp = response.read().decode("utf8")
            resp_obj = json.loads(resp)
            return resp_obj
    except HTTPError as e:
        logger.error(e)


def set_job_success(job):
    url = f'https://{blue_api_base_url}/workflow_tag/job/success/{job["reference_workflow_tag_id"]}'
    request = urllib.request.Request(url=url)
    request.add_header("Content-type", "application/json")
    request.add_header("Accept", "application/json")
    try:
        urllib.request.urlopen(request)
    except HTTPError as e:
        logger.error(e)


def get_file_from_abc_reffile_obj(referencefile_json_obj):
    file_download_api = (f"https://{blue_api_base_url}/reference/referencefile/download_file/"
                         f"{referencefile_json_obj['referencefile_id']}")
    token = get_authentication_token()
    headers = generate_headers(token)
    try:
        response = requests.request("GET", file_download_api, headers=headers)
        return response.content
    except requests.exceptions.RequestException as e:
        logger.info(f"Error occurred for accessing/retrieving data from {file_download_api}: error={e}")
        return None


def download_tei_files_for_references(reference_curies: List[str], output_dir: str, mod_abbreviation):
    for reference_curie in tqdm(reference_curies, desc="Downloading TEI files"):
        all_reffiles_for_pap_api = f'https://{blue_api_base_url}/reference/referencefile/show_all/{reference_curie}'
        request = urllib.request.Request(url=all_reffiles_for_pap_api)
        request.add_header("Content-type", "application/json")
        request.add_header("Accept", "application/json")
        try:
            with urllib.request.urlopen(request) as response:
                resp = response.read().decode("utf8")
                resp_obj = json.loads(resp)
                for ref_file in resp_obj:
                    if ref_file["file_extension"] == "tei" and ref_file["file_class"] == "tei" and any(
                            ref_file_mod["mod_abbreviation"] == mod_abbreviation for ref_file_mod in
                            ref_file["referencefile_mods"]):
                        file_content = get_file_from_abc_reffile_obj(ref_file)
                        with open(os.path.join(output_dir, reference_curie.replace(
                                ":", "_") + ".pdf"), "wb") as out_file:
                            out_file.write(file_content)
        except HTTPError as e:
            logger.error(e)


def download_classification_model(mod_abbreviation: str, topic: str):
    ...


def upload_classification_model(mod_abbreviation: str, topic: str, model_path: str):
    ...
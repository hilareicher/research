# version 18-11-2024  2:33
import argparse
import re
import string
import utils

import requests
import csv
import json
import pathlib
import sys
import os
import meds

from itertools import chain

from age import get_age_entities
from criminal_case import get_criminal_case
# from person import get_names_entities
# from person import get_name_replacements
from orgs import get_hosp_name, get_org_replacements
from partial_date import get_partial_date_entities
from suggestions import suggest_replacement, clean_input

files_errors = []

CategoryMapping = {
    'date': ['DATE', 'DATE_TIME', 'LATIN_DATE', 'NOISY_DATE', 'PREPOSITION_DATE', 'PARTIAL_DATE', 'TIME']}


def extract_additional_entities(text):
    try:
        # cities_entities = get_cities_from_text(text)
        hosp_name = get_hosp_name(text)
        criminal_case = get_criminal_case(text)
        age_entities = get_age_entities(text)
        partial_date_entities = get_partial_date_entities(text)
        all_entities = age_entities + partial_date_entities + criminal_case + hosp_name
        # exclusion_list = get_exclusion_list()
        # # print("Extracted additional entities successfully.")
        # filtered_entities = [entity for entity in all_entities if entity not in exclusion_list]

        return all_entities
    except Exception as e:
        print(f"Error extracting additional entities: {e}")
        return []


def get_context(text, start_position, end_position):
    try:
        start_index = max(0, start_position - 20)
        end_index = min(len(text), end_position + 20)
        context = text[start_index:end_index]
        # print("Context extracted successfully.")
        return context.replace("\n", " . ").replace(",", " ")
    except Exception as e:
        print(f"Error extracting context: {e}")
        return ""


def check_duplicates(prev_text, current_text):
    try:
        for prev in prev_text:
            if abs(prev[2] - current_text[2]) <= 2 or abs(prev[1] - current_text[1]) <= 2:
                if prev[0] in current_text[0] or current_text[0] in prev[0]:
                    # print("Duplicate found.")
                    return True
        return False
    except Exception as e:
        print(f"Error checking duplicates: {e}")
        return False


def extract_and_replace_entities(response, input_dir, skip_type):
    doc_entities = []
    for doc in response.get("docs", []):
        try:
            try:
                with open(os.path.join(input_dir, doc["id"]), 'r', encoding='utf-8') as file:
                    # print(f"***************************************************************.")
                    text, title = clean_text(file.read())
                    # print(f"File {doc['id']} loaded and cleaned successfully.")
                    # print(f"***************************************************************.")
            except FileNotFoundError:
                print(f"File not found: {doc['id']}")
                sys.exit(1)

            additional_items = extract_additional_entities(text)
            prev_text_info = []
            for item in chain(doc.get("items", []), additional_items):
                if item["text"] in string.punctuation:
                    continue  # Skip this item if it is a punctuation mark
                if item["text"] == 'בש' and any(
                        word in get_context(text, item.get("textStartPosition", -1), item.get("textEndPosition", -1))
                        for word in
                        ['בגד', 'בגדים', 'חולצה', 'מכנסיים', 'שמלה', 'חצאית', 'מעיל', 'ג׳קט', 'מכנס', 'כובע',
                         'נעליים']):
                    continue

                if item["text"].isnumeric() and any(
                        word in get_context(text, item.get("textStartPosition", -1), item.get("textEndPosition", -1))
                        for word in ['נפשות', 'אחים', 'נעליים']):
                    continue
                if skip_type and item["textEntityType"] in CategoryMapping.get(skip_type, []):
                    # print(f"Skipping item with type: {item['textEntityType']}")
                    continue
                # item["text"] =clean_input(item["text"])
                if item["text"] in utils.exclusion_list:
                    continue
                if meds.check_meds_match(item["text"]):
                    continue
                replacement = suggest_replacement(item["textEntityType"], item["text"], doc["id"],
                                                  item["maskOperator"], item["mask"])
                if not replacement or replacement['replacement_value'] == item['text']:
                    continue
                ann = {
                    "doc_id": doc["id"],
                    "textStartPosition": item.get("textStartPosition", -1),
                    "original": item["text"],
                    "type": item["textEntityType"],
                    "replacement": replacement["replacement_value"],
                    "context": get_context(text, item.get("textStartPosition", -1), item.get("textEndPosition", -1)),
                    "is_replaced": item["text"] != replacement["replacement_value"],
                    "in_exclusion_list": replacement.get("in_exclusion_list", "False"),
                    # this test is always performed, can't be NA
                    "is_identifying_prefix": replacement.get("is_identifying_prefix", "NA"),
                    "unidentified_subtype": replacement.get("unidentified_subtype", "NA"),
                    "justification": replacement.get("justification", "NA")
                }

                if prev_text_info and item["textEntityType"] in ['ORG', 'PERS', 'PER', 'CITY', 'LOC']:
                    if check_duplicates(prev_text_info,
                                        [item['text'], item['textStartPosition'], item.get("textEndPosition", -1)]):
                        continue
                if item["textEntityType"] in ['ORG', 'PERS', 'PER', 'CITY', 'LOC', 'FAC']:
                    prev_text_info.append([item['text'], item['textStartPosition'], item.get("textEndPosition", -1)])

                doc_entities.append(ann)
            # print(f"***************************************************************.")
            # print(f"Entities extracted and replacements suggested for file {doc['id']}.")
            # print(f"***************************************************************.")
        except Exception as e:
            print(f"Error processing file {doc['id']}: {e}")
    return doc_entities


def send_request_to_server(endpoint, body):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    try:
        # Log the request details
        # print("Sending request to server...")
        # print(f"Endpoint: {endpoint}")
        # print(f"Request body: {json.dumps(body, indent=4)}")  # Log the JSON body being sent

        response = requests.post(endpoint, json=body, headers=headers)

        # Check the response status code
        # print(f"Response received with status code {response.status_code}.")

        # If the response status code is not 200, log the error
        if response.status_code != 200:
            print(f"Error querying Safe Harbor server: {response.status_code}")
            print(f"Response content: {response.content.decode('utf-8')}")  # Log server response for debugging
            sys.exit(1)

        # If the response is okay, return the JSON data
        return response.json()

    except requests.exceptions.RequestException as req_err:
        # Handle any request-related errors (e.g., connection errors)
        print(f"Request error: {req_err}")
        sys.exit(1)
    except Exception as e:
        # Handle any other exceptions
        print(f"Error sending request to server: {e}")
        sys.exit(1)


# def clean_text(text):
#     # Remove prefix that matches the pattern
#     prefix_pattern = r"^.*?:?\s{15,}"
#     # prefix_pattern = r"^.*?:\s{15,}"
#     cleaned_text = re.sub(prefix_pattern, "", text, count=1, flags=re.MULTILINE)
#     if cleaned_text != text:
#         return cleaned_text
#
#     before_double_return_pattern = r"^.*?\n\n"
#     cleaned_text = re.sub(before_double_return_pattern, "", text, count=1, flags=re.DOTALL)
#     return cleaned_text

def clean_text(text):
    removed_parts = []
    # Remove prefix that matches the pattern
    prefix_pattern = r"^.*?:?\s{15,}"
    prefix_match = re.search(prefix_pattern, text, flags=re.MULTILINE)
    if prefix_match:
        removed_parts.append(prefix_match.group())
        cleaned_text = re.sub(prefix_pattern, "", text, count=1, flags=re.MULTILINE)
    else:
        cleaned_text = text

    # Remove text before double line break
    before_double_return_pattern = r"^.*?\r?\n\r?\n"
    # before_double_return_pattern = r"^.*?\n\n"
    double_return_match = re.search(before_double_return_pattern, cleaned_text, flags=re.DOTALL)
    if double_return_match:
        removed_parts.append(double_return_match.group())
        cleaned_text = re.sub(before_double_return_pattern, "", cleaned_text, count=1, flags=re.DOTALL)

    # print(f"Text cleaned successfully. {cleaned_text}, {removed_parts[0]}")
    # print(f"{removed_parts[0]}")

    return cleaned_text, removed_parts[0] if removed_parts else ""


def create_request_body(files):
    global files_errors
    request_body = {"docs": []}
    try:
        for f in files:
            # print(f"***************************************************************.")
            # print(f"*************** File {f} - create_request_body ***************.")
            # print(f"***************************************************************.")

            text = f.read_text(encoding='utf-8')
            text, title = clean_text(text)
            # Check if both text and title are not empty
            if text and title:
                request_body["docs"].append({"id": f.name, "text": text})
            else:
                # print(f"********************** Error - create_request_body **************************.")
                files_errors.append(f.name)
                # print(f"Skipping file '{f.name}' due to empty text or title.")
                # print(f"********************** Error - create_request_body **************************.")

            # request_body["docs"].append({"id": f.name, "text": text})
        print("Request body created successfully.")
        return request_body
    except Exception as e:
        print(f"Error creating request body: {e}")
        return request_body


def load_input_files(input_dir):
    try:
        input_path = pathlib.Path(input_dir)
        files = list(input_path.glob('*'))
        files = [f for f in files if f.name not in ('response.json', 'entities.csv', 'entities.txt')]
        # Sort the files by their name (alphabetically)
        files.sort(key=lambda f: f.name)
        print(f"Loaded {len(files)} input files from {input_dir}.")
        return files
    except Exception as e:
        print(f"Error loading input files from {input_dir}: {e}")
        return []


def save_json_response_to_file(response, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(response, file, indent=4, sort_keys=True, ensure_ascii=False)
        # print(f"Saved Safe Harbor response to file: {filename}")
    except Exception as e:
        print(f"Error saving JSON response to file: {e}")


# def save_entities_as_csv(entities, filename):
#     try:
#         with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
#             writer = csv.writer(csvfile)
#             headers = ['doc_id', 'textStartPosition', 'original', 'type', 'replacement', 'context', 'is_replaced', 'justification', 'is_identifying_prefix']
#             writer.writerow(headers)
#             for entity in entities:
#                 row = [
#                     entity["doc_id"], entity["textStartPosition"], entity["original"], entity["type"],
#                     entity["replacement"], entity["context"], entity["is_replaced"], entity["justification"], entity["is_identifying_prefix"]
#                 ]
#                 writer.writerow(row)
#         print(f"Saved identified entities to file: {filename}")
#     except Exception as e:
#         print(f"Error saving entities to CSV: {e}")
#         sys.exit(1)


def save_entities_as_csv(entities, filename):
    try:
        # Check if the file exists to decide whether to write the header or not
        file_exists = os.path.exists(filename)

        with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)

            # Write header only if the file is being created for the first time
            if not file_exists:
                headers = ['doc_id', 'textStartPosition', 'original', 'type', 'replacement', 'context', 'is_replaced',
                           'justification', 'is_identifying_prefix']
                writer.writerow(headers)

            # Append each entity's details as a new row in the CSV
            for entity in entities:
                row = [
                    entity["doc_id"], entity["textStartPosition"], entity["original"], entity["type"],
                    entity["replacement"], entity["context"], entity["is_replaced"], entity["justification"],
                    entity["is_identifying_prefix"]
                ]
                writer.writerow(row)

        # print(f"Appended identified entities to file: {filename}")
    except Exception as e:
        print(f"Error saving entities to CSV: {e}")
        sys.exit(1)


# def main(input_dir, endpoint):
#     try:
#         files = load_input_files(input_dir)
#         print(f"Loading input files, path: {input_dir}")
#         print(f"Found {len(files)} input files")
#
#         body = create_request_body(files)
#         response = send_request_to_server(endpoint, body)
#         save_json_response_to_file(response, pathlib.Path(input_dir, "response.json"))
#         entities = extract_and_replace_entities(response, input_dir)
#         save_entities_as_csv(entities, pathlib.Path(input_dir, "entities.txt"))
#         print("Processing completed successfully.")
#     except Exception as e:
#         print(f"Error in main execution: {e}")
#         sys.exit(1)


def main(input_dir, endpoint, batch_size=100, start_file_num=1, skip_type=None):
    global files_errors
    try:
        files = load_input_files(input_dir)
        # print(f"Loading input files, path: {input_dir}")
        total_files = len(files)
        # print(f"Found {total_files} input files")

        # Adjust `start_file_num` to index and ensure it's within valid range
        start_file_num = max(1, min(start_file_num, total_files))  # Ensure it's within the bounds
        start_index = start_file_num - 1  # Convert to 0-based index

        # Split files into chunks of `batch_size`, starting from `start_index`
        for i in range(start_index, total_files, batch_size):
            files_errors = []
            batch_files = files[i:i + batch_size]
            last_file_num = min(i + len(batch_files), total_files)
            first_file_name = batch_files[0] if batch_files else "None"
            last_file_name = batch_files[-1] if batch_files else "None"
            current_batch_num = ((i - start_index) // batch_size) + 1
            # print("********************************************************************************")
            # print(
            #     f"Processing batch {current_batch_num}, files '{first_file_name}' to '{last_file_name}'")
            # print(f"Starting from file number {start_file_num}")
            # print("********************************************************************************")

            body = create_request_body(batch_files)
            response = send_request_to_server(endpoint, body)

            # Save the response and entities for each batch
            save_json_response_to_file(response, pathlib.Path(input_dir, "response.json"))

            entities = extract_and_replace_entities(response, input_dir, skip_type)
            save_entities_as_csv(entities, pathlib.Path(input_dir, "entities.txt"))
            print("********************* Batch completed *************************************")
            print(f"Batch completed. Last file number processed: {last_file_num}\n")
            print(f"Skipping files due to empty text or title {files_errors}\n")
            print("********************************************************************************")

        print("Processing completed successfully.")
    except Exception as e:
        print(f"Error in main execution: {e}")
        sys.exit(1)


def load_config(config_file):
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading configuration file {config_file}: {e}")
        sys.exit(1)

if __name__ == '__main__':
    # either specify the configuration file via an environment variable or default to config.json
    config_file = os.environ.get("CONFIG_FILE", "config.json")
    config = load_config(config_file)

    # Load configuration parameters from a config file
    input_dir = config.get("input_dir") or os.environ.get('EMR_INPUT_DIR')
    if not input_dir:
        raise ValueError("Input directory must be specified in config or via the EMR_INPUT_DIR environment variable.")

    endpoint = config.get("endpoint", "http://127.0.0.1:8000/query")
    batch_size = config.get("batch_size", 100)
    start_file_num = config.get("start_file_num", 1)
    skip_type = config.get("skip_type", None)

    print("Starting processing with the following parameters:")
    print(f"Input Directory: {input_dir}")
    print(f"Batch Size: {batch_size}")
    print(f"Start File Number: {start_file_num}")
    print(f"Skip Type: {skip_type}")

    main(input_dir, endpoint, batch_size=batch_size, start_file_num=start_file_num, skip_type=skip_type)

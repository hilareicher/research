import argparse

import requests
import csv
import json
import pathlib
import sys
import os
from itertools import chain

from age import get_age_entities
from suggestions import suggest_replacement


def extract_additional_entities(text):
    return get_age_entities(text)


def get_context(text, start_position):
    # get 20 characters before and after the entity
    context = text[max(0, start_position - 20):start_position + 20]
    return context.replace("\n", " . ").replace(",", " ")


def extract_and_replace_entities(response, input_dir):
    doc_entities = []
    for doc in response.get("docs", []):
        # read original text in file from input_dir, print error if file not found
        try:
            with open(os.path.join(input_dir, doc["id"]), 'r', encoding='utf-8') as file:
                original_text = file.read()
        except FileNotFoundError:
            print(f"File not found: {doc['id']}")
            original_text = ""
        additional_items = extract_additional_entities(original_text)
        for item in chain(doc.get("items", []), additional_items):
            ann = {
                "doc_id": doc["id"],
                "textStartPosition": item.get("textStartPosition", -1),
                "original": item["text"],
                "type": item["textEntityType"],
                "replacement": suggest_replacement(item["textEntityType"], item["text"], doc["id"],
                                                   item["maskOperator"], item["mask"]),
                "context": get_context(original_text, item.get("textStartPosition", -1))
            }
            doc_entities.append(ann)

    return doc_entities


def send_request_to_server(endpoint, body):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    response = requests.post(endpoint, json=body, headers=headers)
    if response.status_code != 200:
        print(f"Error querying Safe Harbor server: {response.status_code}")
        sys.exit(1)
    return response.json()


def create_request_body(files):
    request_body = {"docs": []}
    for f in files:
        try:
            text = f.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                text = f.read_text(encoding='ISO-8859-1')
            except UnicodeDecodeError:
                print(f"Failed to read file {f.name} with UTF-8 and ISO-8859-1 encodings. Skipping file..")
                continue
        request_body["docs"].append({"id": f.name, "text": text})
    return request_body


def load_input_files(input_dir):
    input_path = pathlib.Path(input_dir)
    files = list(input_path.glob('*'))
    files = [f for f in files if f.name not in ('response.json', 'entities.csv', 'entities.txt')]
    return files


def save_json_response_to_file(response, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(response, file, indent=4, sort_keys=True, ensure_ascii=False)
    print(f"Saved Safe Harbor response, file: {filename}")


def save_entities_as_csv(entities, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['doc_id', 'textStartPosition', 'original', 'type', 'replacement', 'context'])
        for entity in entities:
            writer.writerow([entity["doc_id"], entity["textStartPosition"], entity["original"], entity["type"],
                             entity["replacement"], entity["context"]])
    print(f"Saved identified entities, file: {filename}")


def main(input_dir, endpoint):
    files = load_input_files(input_dir)
    print(f"Loading input files, path: {input_dir}")
    print(f"Found {len(files)} input files")

    body = create_request_body(files)
    response = send_request_to_server(endpoint, body)
    save_json_response_to_file(response, pathlib.Path(input_dir, "response.json"))

    entities = extract_and_replace_entities(response, input_dir)
    save_entities_as_csv(entities, pathlib.Path(input_dir, "entities.txt"))
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process EMR data.')
    parser.add_argument('input_dir', nargs='?', type=str, help='The EMR input directory')
    args = parser.parse_args()
    input_dir = args.input_dir
    if input_dir is None:
        input_dir = os.environ.get('EMR_INPUT_DIR')
    if input_dir is None:
        raise ValueError(
            "EMR input directory must be specified either as a command-line argument or through the EMR_INPUT_DIR environment variable.")
    endpoint = os.environ.get('SAFE_HARBOR_ENDPOINT', "http://127.0.0.1:8000/query")
    main(input_dir, endpoint)

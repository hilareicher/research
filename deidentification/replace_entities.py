import argparse
import csv
import json
import os
from collections import defaultdict
from extract_entities import clean_text


def load_json_from_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def add_indicator(entity):
    return f"**{entity}**"


def load_csv_from_file(file_path):
    with open(file_path) as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def save_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(data, indent=4, ensure_ascii=False))


def save_as_individual_text_files(output_dir, grouped_entities):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    docs_saved = 0
    for doc_id, data in grouped_entities.items():
        file_path = os.path.join(output_dir, doc_id)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(data["new_text"])
        docs_saved += 1

    print(f"Saved {docs_saved} docs")


def load_original_texts(input_dir, grouped_entities):
    for doc_id, data in grouped_entities.items():
        doc_file_path = os.path.join(input_dir, doc_id)
        with open(doc_file_path, 'r', encoding='utf-8') as file:
            original_text = clean_text(file.read())
            data["original_text"] = original_text


def replace_entities_in_response(grouped_entities):
    for doc_id, data in grouped_entities.items():
        print(f"Processing Document ID: {doc_id}")

        replacements = []
        group_text = data["original_text"]

        for entity in data["entities"]:
            if entity["original"] != entity["replacement"]:
                mask = add_indicator(entity["replacement"])
                replacements.append({
                    "mask": mask,
                    "textStartPosition": int(entity["textStartPosition"]),
                    "textEndPosition": int(entity["textStartPosition"]) + len(entity["original"])
                })

        replacements.sort(key=lambda x: -1 * x["textStartPosition"])  # reverse order

        for replacement in replacements:
            start = replacement["textStartPosition"]
            end = replacement["textEndPosition"]
            group_text = group_text[:start] + replacement["mask"] + group_text[end:]

        data["new_text"] = group_text


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('input_dir', help="Directory containing the original text files and entities.txt")
    parser.add_argument('output_dir', help="Directory to save the de-identified text files")

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir

    # if input_dir.endswith("_ORG"):
    #     output_dir = input_dir[:-4] + "_DE-ID"

    entities_file_path = os.path.join(input_dir, "entities.txt")
    print(f"Loading identified entities from file: {entities_file_path}")

    entities = load_csv_from_file(entities_file_path)
    print(f"Found {len(entities)} entities")

    grouped_entities = defaultdict(lambda: {"entities": []})
    for entity in entities:
        entity["textStartPosition"] = int(entity["textStartPosition"])
        entity["is_replaced"] = entity["is_replaced"].lower() == 'true'
        entity["is_identifying_prefix"] = entity["is_identifying_prefix"].lower() == 'true'

        doc_id = entity["doc_id"]
        grouped_entities[doc_id]["entities"].append(entity)

    load_original_texts(input_dir, grouped_entities)

    print("Generating docs with replacements")
    replace_entities_in_response(grouped_entities)

    print(f"Saving anonymized docs to path: {output_dir}")
    save_as_individual_text_files(output_dir, grouped_entities)

    print("Done")


if __name__ == "__main__":
    main()

# usage: deidentification/replace_entities.py [-h] [input_dir]
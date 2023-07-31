import glob, json, pathlib, csv
import os

import requests


# load json from file
def load_json_from_file(file):
    with open(file, 'r') as file:
        return json.load(file)


def replace_entities_in_response():
    for doc in response["docs"]:
        doc["new_text"] = doc["text"]
        doc["marked_new_text"] = doc["text"]
        replacements = []  # list of replacements for this doc
        for item in doc["items"]:
            for entity in entities:
                if entity[0] == doc["id"] and entity[1] == item["text"] \
                        and entity[2] == item["textEntityType"]:
                    # generate text with replaced entity
                    # if no replacement was specified, use the original text
                    mask = entity[3] if len(entity[3]) > 0 else entity[1]
                    is_mask_specified = len(entity[3]) > 0
                    replacements.append({"mask": mask, "maskStartPosition": item["maskStartPosition"],
                                         "maskEndPosition": item["maskEndPosition"],
                                         "is_mask_specified": is_mask_specified})
        # apply replacements to text
        replacements.sort(key=lambda x: -1 * x["maskStartPosition"])  # reverse order
        for replacement in replacements:
            doc["new_text"] = doc["new_text"][:replacement["maskStartPosition"]] + \
                              replacement["mask"] + doc["new_text"][replacement["maskEndPosition"]:]

            css_class = "replaced" if replacement["is_mask_specified"] else "orig"
            doc["marked_new_text"] = doc["marked_new_text"][:replacement["maskStartPosition"]] + \
                                     "<span class=\"" + css_class + "\">" + replacement["mask"] + "</span>" + \
                                     doc["marked_new_text"][replacement["maskEndPosition"]:]
        # print("updated text for doc " + doc["id"] + ": " + doc['new_text'])


def save_to_file(response, f):
    with open(f, 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, indent=4, ensure_ascii=False))


def load_csv_from_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        return list(reader)


def save_as_individual_text_files(response, output_dir):
    docs = 0
    for doc in response["docs"]:
        filename = doc["id"]
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
            file.write(doc["new_text"])
        docs += 1
    print("saved " + str(docs) + " docs")


def save_as_individual_html_files(response, output_dir):
    docs = 0
    for doc in response["docs"]:
        filename = doc["id"] + ".html"
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as file:
            file.write("<html><head><style>"
                       ".orig { color: blue; }"
                       ".replaced { color: red; }"
                       "</style><meta charset=\"UTF-8\"></head><body>")
            file.write(doc["marked_new_text"])
            file.write("</body></html>")
        docs += 1
    print("saved " + str(docs) + " html docs")


# load safe harbor response and entities file
input_dir = os.environ['EMR_INPUT_DIR']
response_file = os.path.join(input_dir, "response.json")
print("loading Safe Harbor response, file: " + response_file)
response = load_json_from_file(response_file)

entities_file = os.path.join(input_dir, "entities.csv")
print("loading identified entities, file: " + entities_file)
entities = load_csv_from_file(entities_file)
print("found " + str(len(entities)) + " entities")

# perform replacements
print("generating docs with replacements")
replace_entities_in_response()

# save in individual files
if input_dir.endswith("_ORG"):
    output_dir = input_dir[:-4] + "_DE-ID"
else:
    output_dir = os.environ['EMR_OUTPUT_DIR']

# create output dir if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# save response with replaced entities
response_file = os.path.join(output_dir, "response_with_replacements.json")
save_to_file(response, response_file)
print("saved anonymized docs, file: " + response_file)

print("saving anonymized docs, path: " + output_dir)
# save_as_individual_text_files(response, output_dir)
save_as_individual_html_files(response, output_dir)

print("done")

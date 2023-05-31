import glob, json, pathlib, csv
import requests


# load json from file
def load_json_from_file(file):
    with open(file, 'r') as file:
        return json.load(file)


def replace_entities_in_response():
    for doc in response["docs"]:
        doc["new_text"] = doc["text"]
        replacements = []  # list of replacements for this doc
        for item in doc["items"]:
            for entity in entities:
                if entity[0] == doc["id"] and entity[1] == item["text"] \
                        and entity[2] == item["textEntityType"]:
                    # generate text with replaced entity
                    # if no replacement was specified, use the original text
                    mask = entity[3] if len(entity[3]) > 0 else entity[1]
                    replacements.append({"mask": mask, "maskStartPosition": item["maskStartPosition"], "maskEndPosition": item["maskEndPosition"]})
        # apply replacements to text
        replacements.sort(key=lambda x: -1*x["maskStartPosition"]) # reverse order
        for replacement in replacements:
            doc["new_text"] = doc["new_text"][:replacement["maskStartPosition"]] + \
                              replacement["mask"] + doc["new_text"][replacement["maskEndPosition"]:]
        # print("updated text for doc " + doc["id"] + ": " + doc['new_text'])

def save_to_file(response, f):
    with open(f, 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, indent=4, ensure_ascii=False))


def load_csv_from_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        return list(reader)


# load safe hrbor response and entities file
print("loading Safe Harbor response, file: response.json")
response = load_json_from_file("response.json")
print("loading identified entities, file: entities.csv")
entities = load_csv_from_file("entities.csv")
print("found " + str(len(entities)) + " entities")

# perform replacements
print("generating docs with replacements")
replace_entities_in_response()

# save response with replaced entities
save_to_file(response, "response_with_replacements.json")
print("saved anonymized docs, file: response_with_replacements.json")
print("done")

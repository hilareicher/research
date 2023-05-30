import glob, json, pathlib
import requests


# load response from file
def load_from_file(file):
    with open(file, 'r') as file:
        return json.load(file)


def replace_names_in_response(response, names):
    for doc in response["docs"]:
        doc["new_text"] = doc["text"]
        for item in doc["items"]:
            if item["textEntityType"] == "PERS":
                for rep in names:
                    if rep['doc_id'] == doc["id"] and rep['orig_name'] == item["text"]:
                        new_text = doc["new_text"][:item["maskStartPosition"]] + \
                                   rep["replace_name"] + doc["text"][item["maskEndPosition"]:]
                        doc["new_text"] = new_text

def save_to_file(response, f):
    with open(f, 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, indent=4, ensure_ascii=False))

response = load_from_file("response.json")
names = load_from_file("orig_names.json")
replace_names_in_response(response, names)
print(json.dumps(response, indent=4, ensure_ascii=False))
save_to_file(response, "response_with_names.json")

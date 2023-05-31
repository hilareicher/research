import glob, json, pathlib, os, sys, requests, csv


# extract entities from response
def extract_entities():
    if "docs" in response:
        docs = response["docs"]
        doc_names = []
        for doc in docs:
            if "items" in doc:
                items = doc["items"]
                for item in items:
                    #  if "textEntityType" in item and item["textEntityType"] == "PERS":
                    ann = {"doc_id": doc["id"], "original": item["text"], "type": item["textEntityType"]}
                    doc_names.append(ann)
        return doc_names


# send POST request to server and return body
def send_request_to_server():
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    res = requests.post(endpoint, data=body, headers=headers)
    if res.status_code != 200:
        print("Error querying Safe Harbor server: " + str(res.status_code))
        sys.exit
    return res.json()


# go over files and create request body
def create_request_body():
    request_body = {}
    arr = []
    for f in files:
        with open(f, 'r') as file:
            text = file.read()
            obj = {"id": f.name, "text": text}
            arr.append(obj)
    request_body["docs"] = arr
    return json.dumps(request_body, indent=4)


# get list of file names in input dir
def load_input_files(input_dir):
    return list(pathlib.Path(input_dir).glob('*'))


# save response to file
def save_json_response_to_file(filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(json.dumps(response, indent=4, sort_keys=True, ensure_ascii=False))


def save_entities_as_csv(filename):
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['doc_id', 'original', 'type', 'replacement'])
        for entity in entities:
            writer.writerow([entity["doc_id"], entity["original"], entity["type"], None])


# load input files
input_dir = os.environ['EMR_INPUT_DIR']
print("loading input files, path: " + input_dir)
files = load_input_files(input_dir)
print("found " + str(len(files)) + " input files")

# send request to Safe Harbor
body = create_request_body()
endpoint = os.environ.get('SAFE_HARBOR_ENDPOINT', "http://127.0.0.1:8000/query")
response = send_request_to_server()
save_json_response_to_file("response.json")
print("saved Safe Harbor response, file: response.json")

# extract entities from response and save to file
entities = extract_entities()
save_entities_as_csv("entities.csv")
print("saved identified entities, file: entities.csv")
print("done")

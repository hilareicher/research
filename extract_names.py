import glob, json, pathlib
import requests


# extract names from response
def extract_names(response):
    response = json.loads(response)
    if "docs" in response:
        docs = response["docs"]
        doc_names = []
        for doc in docs:
            if "items" in doc:
                items = doc["items"]
                for item in items:
                    if "textEntityType" in item and item["textEntityType"] == "PERS":
                        pers = {"doc_id": doc["id"], "orig_name": item["text"]}
                        doc_names.append(pers)
        return doc_names


# send POST request to server and return body
def send_request_to_server(body):
    url = "http://127.0.0.1:8000/query"
    headers = {"Content-Type": "application/json",
               "Accept": "application/json"}
    response = requests.post(url, data=body, headers=headers)
    if response.status_code != 200:
        print("Error: " + str(response.status_code))
    return response.content.decode("utf-8")


# go over files and create request body
def create_request_body(files):
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
def save_json_to_file(json, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(json)


input_dir = "/Users/hilac/EMRs"
files = load_input_files(input_dir)
print(files)
body = create_request_body(files)
print(body)
response = send_request_to_server(body)
save_json_to_file(response, "response.json")
names = extract_names(response)
names.sort(key=lambda x: x["doc_id"])
print(names)
save_json_to_file(json.dumps(names, ensure_ascii=False, indent=4), "orig_names.json")

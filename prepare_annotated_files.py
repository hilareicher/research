import os
import shutil
import zipfile


project_dir = os.environ['INCEPTION_PROJECT_DIR']
if not os.path.isdir(project_dir):
    print("Error: INCEPTION_PROJECT_DIR is not a directory")
    exit(1)
print("reading INCEpTION project directory, path: " + project_dir)

print("creating annotated_output directory under project directory")
output_dir = os.path.join(project_dir, "annotated_output")
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

annotation_dir = os.path.join(project_dir, "annotation")
if not os.path.isdir(annotation_dir):
    print("Error: annotation directory not found in project directory")
    exit(1)

docs = {}
for dir_name in os.listdir(annotation_dir):
    if dir_name not in docs:
        docs[dir_name] = {}
print("found " + str(len(docs)) + " document ids in annotation directory")

for doc_id in docs:
    doc_dir = os.path.join(annotation_dir, doc_id)
    if not os.path.isdir(doc_dir):
        print(
            "Error: directory " + doc_dir + " not found under the project's annotation directory, skipping document...")
        continue

    # each annotating user should have a zip file, and inside the zip an XMI file with the name: <username>.xmi

    for dir_name in os.listdir(doc_dir):
        if dir_name.endswith(".zip"):
            print("processing directory: " + dir_name[:-4])
            # unzip annotator zip file to a directory called <username>
            annotator_dir = os.path.join(doc_dir, dir_name[:-4])
            annotator_zip_file = os.path.join(doc_dir, dir_name)
            with zipfile.ZipFile(annotator_zip_file, 'r') as zip_ref:
                zip_ref.extractall(annotator_dir)

            # find the XMI file in the annotator directory, there should only be one XMI file
            for f in os.listdir(annotator_dir):
                if f.endswith(".xmi"):
                    xmi_file = os.path.join(annotator_dir, f)
                    break
            if not xmi_file:
                print(
                    "Error: XMI file not found in " + dir_name + " directory " + annotator_dir + ", skipping ...")
                continue

            # copy and rename file to output directory. The new name of the xmi file should be it's parent directory name
            output_file = os.path.join(output_dir, doc_id)
            print (f"copying file from {xmi_file} to {output_file}")
            shutil.copyfile(xmi_file, output_file)


print ("done!")


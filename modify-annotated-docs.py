# This script modifies documents annotated in INCEpTION according to an external file containing replacements.
# The replacements file is in the following format:
# doc_id, current_value, replacement_value
import csv
import os
import zipfile
from cassis import *

# In addition, the script will perform three additional tasks:
# 1) remove <_יום> tags
# 2) replace <_קשר> tags with a random number of 10 digits
# 3) remove ** from the beginning and end of words

# the project directory is the directory containing the exported INCEpTION project
# the project directory should be specified in an environment variable called INCEPTION_PROJECT_DIR
# The script will output a new file with the same name as the input file, with the suffix "_modified"


def load_csv_from_file(file):
    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        return list(reader)


def parse_replacements_file(f):
    # read the replacements file, group lines by doc_id, and create a dictionary of replacements
    replacements = {}
    replacements_list = load_csv_from_file(f)
    for line in replacements_list:
        doc_id = line[0].strip()
        current_value = line[1].strip()
        replacement_value = line[2].strip()
        if doc_id not in replacements:
            replacements[doc_id] = {}
        replacements[doc_id][current_value] = replacement_value
    return replacements


def apply_replacements(text, replacements_map):
    for orig_value in replacements_map:
        text = text.replace(orig_value, replacements_map[orig_value])
    return text


project_dir = os.environ['INCEPTION_PROJECT_DIR']
if not os.path.isdir(project_dir):
    print("Error: INCEPTION_PROJECT_DIR is not a directory")
    exit(1)
print("loading INCEpTION project directory, path: " + project_dir)
# for each doc_id, we search for it in the project directory
# if found, we modify it according to the replacements file
# if not found, we skip it

replacements_file = os.environ['REPLACEMENTS_FILE']
replacements = parse_replacements_file(replacements_file)
print("replacements file contains " + str(len(replacements)) + " document ids")

# under 'annotation' directory in the project directory there should
# be a directory for each doc_id
annotation_dir = os.path.join(project_dir, "annotation")
if not os.path.isdir(annotation_dir):
    print("Error: annotation directory not found in project directory")
    exit(1)

# add documents that are present in the annotation directory but not in the replacements file
for dir_name in os.listdir(annotation_dir):
    doc_id = dir_name[:-4]
    if doc_id not in replacements:
        replacements[doc_id] = {}
print("found " + str(len(replacements)) + " document ids in annotation directory")
for doc_id in replacements:
    replacements[doc_id]["&lt;_יום&gt"] = ""
    replacements[doc_id]["&lt;_קשר&gt"] = "1234567890"
    replacements[doc_id]["**"] = ""

for doc_id in replacements:
    doc_dir = os.path.join(annotation_dir, doc_id + ".txt")
    if not os.path.isdir(doc_dir):
        print(
            "Error: directory " + doc_dir + " not found under the project's annotation directory, skipping document...")
        continue

    # each annotating user should have a zip file, and inside the zip an XMI file with the name: <username>.xmi

    # for each annotater, modify text, shift annotations, and write to a new XMI file
    # the new XMI file should be called <username>_MODIFIED.xmi
    for dir_name in os.listdir(doc_dir):
        if dir_name.endswith(".zip"):
            print("processing directory: " + dir_name[:-4])
            # unzip annotator zip file to a directory called <username>
            annotator_dir = os.path.join(doc_dir, dir_name[:-4])
            annotator_zip_file = os.path.join(doc_dir, dir_name)
            with zipfile.ZipFile(annotator_zip_file, 'r') as zip_ref:
                zip_ref.extractall(annotator_dir)

            # load XMI file
            # find the XMI file in the annotator directory, there should only be one XMI file
            for f in os.listdir(annotator_dir):
                if f.endswith(".xmi") and not f.endswith("_MODIFIED.xmi"):
                    xmi_file = os.path.join(annotator_dir, f)
                    break
            if not xmi_file:
                print("Error: XMI file not found in annotator directory " + annotator_dir + ", skipping annotator...")
                continue
            xml_file = os.path.join(annotator_dir, "TypeSystem.xml")
            print("loading XMI file: " + xmi_file)
            with open(xml_file, 'rb') as f:
                typesystem = load_typesystem(f)
            with open(xmi_file, 'rb') as f:
                cas = load_cas_from_xmi(f, typesystem=typesystem)
            annotator_username = os.path.basename(xmi_file)
            # modify the text and shift annotations
            modified_text = cas.sofa_string
            for current_value in replacements[doc_id]:
                replacement_value = replacements[doc_id][current_value]
                modified_text = modified_text.replace(current_value, replacement_value)

            original_text = cas.sofa_string

            # we loop over annotation types defined in the type system:
            for t in typesystem.get_types():
                if 'custom' in t.name:  # we are just interested in the custom annotation types
                    type_annotations = cas.select(t.name)
                    if len(type_annotations) == 0:
                        continue
                    print ("processing annotation type: " + t.name + " with " + str(len(cas.select(t.name))) + " annotations in document " + doc_id + " for annotator " + annotator_username)
                    for annotation in type_annotations:
                        # get the annotation text
                        original_annotation_text = original_text[annotation.begin:annotation.end]
                        expected_modified_text = apply_replacements(original_annotation_text, replacements[doc_id])
                        # find the modified annotation text in the modified text
                        annotation_begin = modified_text.find(expected_modified_text)
                        annotation_end = annotation_begin + len(expected_modified_text)
                        # if the annotation text was found, re-define the annotation
                        #print("replacing " + original_annotation_text + " with " + modified_text[
                        #                                                           annotation_begin:annotation_end] + " in document " + doc_id + " for annotator " + xmi_file[
                        #                                                                                                                                             :-4])
                        if annotation_begin != -1:
                            annotation.begin = annotation_begin
                            annotation.end = annotation_end

                        else:
                            print("Error: annotation text not found in modified text, skipping annotation...")
                            continue

            cas.sofa_string = modified_text
            # # write the modified XMI file to <username>_MODIFIED.xmi
            xmi_file_modified = os.path.join(annotator_dir, xmi_file[:-4] + "_MODIFIED.xmi")
            print("writing modified XMI file: " + xmi_file_modified)
            cas.to_xmi(xmi_file_modified, pretty_print=True)

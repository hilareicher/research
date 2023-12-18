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


def parse_replacements_file():
    # read the replacements file, group lines by doc_id, and create a dictionary of replacements
    replacements = {}
    replacements_file = "replacements.csv"
    replacements_list = load_csv_from_file(replacements_file)
    for line in replacements_list:
        doc_id = line[0]
        current_value = line[1]
        replacement_value = line[2]
        if doc_id not in replacements:
            replacements[doc_id] = {}
        replacements[doc_id][current_value] = replacement_value
    return replacements


project_dir = os.environ['INCEPTION_PROJECT_DIR']
if not os.path.isdir(project_dir):
    print("Error: INCEPTION_PROJECT_DIR is not a directory")
    exit(1)
print("loading INCEpTION project directory, path: " + project_dir)
# for each doc_id, we search for it in the project directory
# if found, we modify it according to the replacements file
# if not found, we skip it

replacements_file = os.environ['REPLACEMENTS_FILE']
replacements = parse_replacements_file()
print("replacements file contains " + str(len(replacements)) + " document ids")
print("adding <_יום> and <_קשר> tags to replacements map")
for doc_id in replacements:
    replacements[doc_id]["<_יום>"] = ""
    replacements[doc_id]["<_קשר>"] = "1234567890"
    replacements[doc_id]["**"] = ""

# under 'annotation' directory in the project directory there should
# be a directory for each doc_id
annotation_dir = os.path.join(project_dir, "annotation")
if not os.path.isdir(annotation_dir):
    print("Error: annotation directory not found in project directory")
    exit(1)
skipped_docs = 0


def apply_replacements(text, replacements_map):
    for orig_value in replacements_map:
        text = text.replace(orig_value, replacements_map[orig_value])
    return text


for doc_id in replacements:
    doc_dir = os.path.join(annotation_dir, doc_id)
    if not os.path.isdir(doc_dir):
        print(
            "Error: directory " + doc_id + " not found under the project's annotation directory, skipping document...")
        skipped_docs += 1
        continue

    # each annotating user should have a zip file with the name: <username>.zip,
    # and an additional canonical zip file with the name: INITIAL_CAS.zip

    # INITIAL_CASE.zip contains the original document text without annotations.
    # unzip it, load XMI using DKPro library, modify the text according to the replacements dictionary

    # unzip INITIAL_CAS.zip to a directory called INITIAL_CAS
    canonical_zip_file = os.path.join(doc_dir, "INITIAL_CAS.zip")
    if not os.path.isfile(canonical_zip_file):
        print("Error: INITIAL_CAS.zip not found in document directory " + doc_dir + ", skipping document...")
        skipped_docs += 1
        continue
    with zipfile.ZipFile(canonical_zip_file, 'r') as zip_ref:
        zip_ref.extractall(doc_dir + "/INITIAL_CAS")

    # load XMI file
    xmi_file = os.path.join(doc_dir, "INITIAL_CAS", "INITIAL_CAS.xmi")
    xml_file = os.path.join(doc_dir, "INITIAL_CAS", "TypeSystem.xml")
    if not os.path.isfile(xmi_file):
        print("Error: INITIAL_CAS.xmi not found in document directory " + doc_dir + ", skipping document...")
        skipped_docs += 1
        continue
    print("loading XMI file: " + xmi_file)
    with open(xml_file, 'rb') as f:
        typesystem = load_typesystem(f)
    with open(xmi_file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem)

    # modify the text
    text = cas.sofa_string
    for current_value in replacements[doc_id]:
        print("replacing " + current_value + " with " + replacements[doc_id][
            current_value] + " in document " + doc_id + " for INITIAL_CAS")
        replacement_value = replacements[doc_id][current_value]
        text = text.replace(current_value, replacement_value)
    cas.sofa_string = text

    # write the modified XMI file to INITIAL_CAS_MODIFIED.xmi
    xmi_file_modified = os.path.join(doc_dir, "INITIAL_CAS", "INITIAL_CAS_MODIFIED.xmi")
    print("writing modified XMI file: " + xmi_file_modified)
    cas.to_xmi(xmi_file_modified)

    # for each annotater, modify text, shift annotations, and write to a new XMI file
    # the new XMI file should be called <username>_MODIFIED.xmi
    for annotator in os.listdir(doc_dir):
        if annotator.endswith(".zip") and annotator != "INITIAL_CAS.zip":
            print("processing annotator: " + annotator[:-4])
            # unzip annotator zip file to a directory called <username>
            annotator_dir = os.path.join(doc_dir, annotator[:-4])
            annotator_zip_file = os.path.join(doc_dir, annotator)
            with zipfile.ZipFile(annotator_zip_file, 'r') as zip_ref:
                zip_ref.extractall(annotator_dir)

            # load XMI file
            xmi_file = os.path.join(annotator_dir, annotator[:-4] + ".xmi")
            xml_file = os.path.join(annotator_dir, "TypeSystem.xml")
            if not os.path.isfile(xmi_file):
                print("Error: " + annotator[
                                  :-4] + ".xmi not found in annotator directory " + annotator_dir + ", skipping annotator...")
                continue
            print("loading XMI file: " + xmi_file)
            with open(xml_file, 'rb') as f:
                typesystem = load_typesystem(f)
            with open(xmi_file, 'rb') as f:
                cas = load_cas_from_xmi(f, typesystem=typesystem)

            # modify the text and shift annotations
            modified_text = cas.sofa_string
            for current_value in replacements[doc_id]:
                replacement_value = replacements[doc_id][current_value]
                modified_text = modified_text.replace(current_value, replacement_value)

            original_text = cas.sofa_string
            # go over annotations of type "webanno.custom.General" and re-define them according to the new text

            for annotation in cas.select("webanno.custom.General"):
                print(annotation)
                # get the annotation text
                original_annotation_text = original_text[annotation.begin:annotation.end]
                expected_modified_text = apply_replacements(original_annotation_text, replacements[doc_id])
                # find the modified annotation text in the modified text
                annotation_begin = modified_text.find(expected_modified_text)
                annotation_end = annotation_begin + len(expected_modified_text)
                # if the annotation text was found, re-define the annotation
                print("replacing " + original_annotation_text + " with " + modified_text[annotation_begin:annotation_end] + " in document " + doc_id + " for annotator " + annotator[:-4])
                if annotation_begin != -1:
                    annotation.begin = annotation_begin
                    annotation.end = annotation_end

                else:
                    print("Error: annotation text not found in modified text, skipping annotation...")
                    continue

            cas.sofa_string = modified_text
            # # write the modified XMI file to <username>_MODIFIED.xmi
            xmi_file_modified = os.path.join(annotator_dir, annotator[:-4] + "_MODIFIED.xmi")
            print("writing modified XMI file: " + xmi_file_modified)
            cas.to_xmi(xmi_file_modified)

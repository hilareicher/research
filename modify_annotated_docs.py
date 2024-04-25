# This script modifies documents annotated in INCEpTION according to an external file containing replacements.
# The replacements file is in the following format:
# doc_id, current_value, replacement_value
import csv
import os
import re
import zipfile

from cassis.xmi import CasXmiSerializer

from cassis import *

NAMESPACE_SEPARATOR = "."
NAME_SPACE_UIMA_TCAS = "uima" + NAMESPACE_SEPARATOR + "tcas"
UIMA_TCAS_PREFIX = NAME_SPACE_UIMA_TCAS + NAMESPACE_SEPARATOR
TYPE_NAME_ANNOTATION = UIMA_TCAS_PREFIX + "Annotation"

# tokens to remove
tokens_remove = ["*",
                 "קשר",
                 "יום",
                 "<", ">", "_"]


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
        return list(reader)


def parse_replacements_file(f):
    # read the replacements file, group lines by doc_id, and create a dictionary of replacements
    replacements = {}
    replacements_list = load_csv_from_file(f)
    for line in replacements_list:
        if (len(line) == 0):
            continue
        doc_id = line[0].strip()
        current_value = line[1].strip()
        replacement_value = line[2].strip()
        if doc_id not in replacements:
            replacements[doc_id] = {}
        replacements[doc_id][current_value] = replacement_value
    return replacements


def apply_replacements(text, replacements_map):
    for orig_value in replacements_map:
        text = selective_replace(text, orig_value, replacements_map[orig_value])
    return text


def selective_replace(string, target, replacement):
    pattern = re.escape(target)
    # Find all matches
    matches = re.findall(pattern, string)
    # if more than one match and target is not one of the special generic tasks then notify
    if len(matches) > 1 and target != "**" and target != "<_קשר>" and target != "<_יום>":
        print("found " + str(len(matches)) + " matches to replace " + target)

    replaced_string = re.sub(pattern, replacement, string)
    return replaced_string


def execute_modification_in_dir(dir_name):
    annotation_dir = os.path.join(project_dir, dir_name)
    if not os.path.isdir(annotation_dir):
        print("Error: " + dir_name +" directory not found in project directory")
        exit(1)

    global fs
    # add documents that are present in the annotation directory but not in the replacements file
    for dir_name in os.listdir(annotation_dir):
        doc_id = dir_name[:-4]
        if doc_id not in replacements:
            replacements[doc_id] = {}
    print("found " + str(len(replacements)) + " document ids in annotation directory")
    for doc_id in replacements:
        replacements[doc_id]["<_יום>"] = ""
        replacements[doc_id]["<_קשר>"] = "1234567890"
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
                    print(
                        "Error: XMI file not found in " + dir_name + " directory " + annotator_dir + ", skipping ...")
                    continue
                xml_file = os.path.join(annotator_dir, "TypeSystem.xml")
                print("loading XMI file: " + xmi_file)
                with open(xml_file, 'rb') as f:
                    typesystem = load_typesystem(f)
                with open(xmi_file, 'rb') as f:
                    cas = load_cas_from_xmi(f, typesystem=typesystem)
                annotator_username = os.path.basename(xmi_file)
                # modify the text and shift annotations
                modified_text = cas.get_sofa().sofaString
                modified_text = apply_replacements(modified_text, replacements[doc_id])

                original_text = cas.get_sofa().sofaString
                # print length of original text

                for fs in sorted(cas._find_all_fs(), key=lambda a: a.xmiID):
                    t = fs.type
                    # check if we have 'begin' and 'end' features
                    if hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0:
                        original_fs_text = fs.get_covered_text()
                        expected_modified_text = apply_replacements(original_fs_text, replacements[doc_id])
                        # find the modified annotation text in the modified text
                        fs_modified_begin = modified_text.find(expected_modified_text)
                        fs_modified_end = fs_modified_begin + len(expected_modified_text)

                        # if the text was found, re-define the object's begin and end
                        # print("replacing " + original_annotation_text + " with " + modified_text[
                        #                                                           fs_modified_begin:fs_modified_end] + " in document " + doc_id + " for annotator " + xmi_file[
                        #                                                                                                                                               :-4])

                        if fs_modified_begin != -1:
                            fs.begin = fs_modified_begin
                            fs.end = fs_modified_end

                        else:
                            # if Token type AND one of the chars that are removed: *, _, <, >
                            if (".Token" in fs.type.name) or "*" in original_fs_text:
                                cas.remove(fs)
                                continue

                            print("Error: annotation text not found in modified text, skipping ...")
                            print(
                                "For debug: " + "type: " + fs.type.name + ", expected modified text: [" + expected_modified_text + "]")
                            print("original_fs_text: [" + original_fs_text + "]")
                            continue

                # replace sofa string with modified text
                cas.get_sofa().sofaString = modified_text

                # write the modified XMI file to <username>_MODIFIED.xmi
                xmi_file_modified = os.path.join(annotator_dir, xmi_file[:-4] + "_MODIFIED_" + doc_id + ".xmi")
                print("writing modified XMI file: " + xmi_file_modified)
                cas.to_xmi(xmi_file_modified, pretty_print=True)


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

print ("1# going to modify annotation directory")
execute_modification_in_dir("annotation")
print ("2# going to modify curation directory")
execute_modification_in_dir("curation")

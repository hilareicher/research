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
    result_text = ""
    change_list = [0] * (len(text) + 1)
    current_index = 0

    # Collect and apply replacements
    changes = []
    for target, repl in replacements_map.items():
        for match in re.finditer(re.escape(target), text):
            start, end = match.start(), match.end()
            # Check if target is surrounded by ** and replacement is not surrounded by **
            if start >= 2 and text[start - 2:start] == "**" and end <= len(text) - 2 and text[end:end + 2] == "**":
                if "**" in target:
                    changes.append((start, end, repl))
            else:
                changes.append((start, end, repl))
    changes.sort()

    for start, end, repl in changes:
        result_text += text[current_index:start] + repl
        offset_change = len(repl) - (end - start)
        for i in range(end, len(text) + 1):
            if i < len(change_list):
                change_list[i] += offset_change
        current_index = end
    result_text += text[current_index:]

    return result_text, change_list


def remove_patterns(text, annotations, patterns):
    final_text = ""
    change_list = [0] * (len(text) + 1)
    current_index = 0

    for pattern in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            final_text += text[current_index:start]
            offset_change = -(end - start)
            for i in range(end, len(text) + 1):
                if i < len(change_list):
                    change_list[i] += offset_change
            current_index = end
        final_text += text[current_index:]
        text = final_text
        final_text = ""
        current_index = 0

    final_text += text[current_index:]
    new_annotations = adjust_annotations(annotations, change_list)
    return final_text, new_annotations


def adjust_annotations(annotations, change_list):
    adjusted_annotations = []
    for start, end in annotations:
        new_start = start + change_list[start]
        new_end = end + (change_list[end] if end < len(change_list) else len(change_list) - 1)
        adjusted_annotations.append((new_start, new_end))
    return adjusted_annotations


def execute_modification_in_dir(dir_name):
    annotation_dir = os.path.join(project_dir, dir_name)
    if not os.path.isdir(annotation_dir):
        print("Error: " + dir_name + " directory not found in project directory")
        exit(1)

    global fs
    # add documents that are present in the annotation directory but not in the replacements file
    for dir_name in os.listdir(annotation_dir):
        dir_name
        doc_id = dir_name[:-4]
        if doc_id not in replacements:
            replacements[doc_id] = {}
    print("found " + str(len(replacements)) + " document ids in annotation directory")
    for doc_id in replacements:
        replacements[doc_id]["<_יום>"] = ""
        replacements[doc_id]["<_קשר>"] = "1234567890"
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
                modified_text, change_list = apply_replacements(cas.get_sofa().sofaString, replacements[doc_id])
                adjusted_annotations = adjust_annotations(
                    [(fs.begin, fs.end) for fs in cas._find_all_fs() if
                     hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0],
                    change_list)

                # adjust annotations directly using the change_list
                for fs in cas._find_all_fs():
                    if hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0:
                        # calculate new begin and end using the change_list
                        new_begin = fs.begin + change_list[fs.begin] if fs.begin < len(change_list) else fs.begin
                        new_end = fs.end + change_list[fs.end] if fs.end < len(change_list) else fs.end
                        fs.begin = new_begin
                        fs.end = new_end

                # replace sofa string with modified text
                cas.get_sofa().sofaString = modified_text

                # Remove ** and prefix pattern, and adjust annotations
                prefix_pattern = r"^.*?:\s{15,}"
                final_text, final_annotations = remove_patterns(cas.get_sofa().sofaString, adjusted_annotations,
                                                                [prefix_pattern])

                for fs, (new_begin, new_end) in zip(cas._find_all_fs(), final_annotations):
                    if hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0:
                        fs.begin = new_begin
                        fs.end = new_end

                cas.get_sofa().sofaString = final_text

                final_text, final_annotations = remove_patterns(cas.get_sofa().sofaString, final_annotations,
                                                                [r'\*\*'])

                cas.get_sofa().sofaString = final_text
                for fs, (new_begin, new_end) in zip(cas._find_all_fs(), final_annotations):
                    if hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0:
                        fs.begin = new_begin
                        fs.end = new_end

                # remove fs that have begin and end larger than the text length
                for fs in cas._find_all_fs():
                    if hasattr(fs, "begin") and hasattr(fs, "end") and (fs.begin > len(cas.get_sofa().sofaString) or fs.end > len(cas.get_sofa().sofaString)):
                        print ("removing fs with begin: " + str(fs.begin) + " and end: " + str(fs.end) + " of type " + fs.type.name)
                        cas.remove(fs)

                # write the modified XMI file to <username>_MODIFIED.xmi
                xmi_file_modified = os.path.join(annotator_dir, xmi_file[:-4] + "_MODIFIED_" + doc_id + ".xmi")
                print("writing modified XMI file: " + xmi_file_modified)
                cas.to_xmi(xmi_file_modified, pretty_print=True)


project_dir = os.environ['INCEPTION_PROJECT_DIR']
if not os.path.isdir(project_dir):
    print("Error: INCEPTION_PROJECT_DIR is not a directory")
    exit(1)
print("loading INCEpTION project directory, path: " + project_dir)

replacements_file = os.environ['REPLACEMENTS_FILE']
replacements = parse_replacements_file(replacements_file)
print("replacements file contains " + str(len(replacements)) + " document ids")

print("1# going to modify annotation directory")
execute_modification_in_dir("annotation")
print("2# going to modify curation directory")
execute_modification_in_dir("curation")

# -----------
project_dir = "/Users/hilac"
replacements_file = os.environ['REPLACEMENTS_FILE']
replacements = parse_replacements_file(replacements_file)

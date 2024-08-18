import csv
import math
import os
import random
from cassis import *
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sympy.physics.control.control_plots import plt

DEBUG = False
iou_threshold = 1  # exact match


def extract_labels_partial_match(annotations_a, annotations_b, all_labels_a, all_labels_b):
    labels_a = []
    labels_b = []
    # Check each annotation in A against all in B for overlap
    for a in annotations_a:
        matched = False
        for b in annotations_b:
            if calculate_iou((a[0], a[1]), (b[0], b[1])) >= iou_threshold:
                labels_a.append(a[2])  # Assuming the format is (start, end, label)
                labels_b.append(b[2])
                matched = True
                break
        if not matched:
            labels_a.append(a[2])
            labels_b.append("O")

    # Repeat for annotations_b to catch any that weren't matched to annotations_a
    for b in annotations_b:
        if not any(calculate_iou((b[0], b[1]), (a[0], a[1])) >= iou_threshold for a in annotations_a):
            labels_b.append(b[2])
            labels_a.append("O")
    # add labels to the global list
    all_labels_a.extend(labels_a)
    all_labels_b.extend(labels_b)
    return labels_a, labels_b


# Intersection Over Union (IoU) for spans
def calculate_iou(span_a, span_b):
    """Calculate the IoU of two spans."""
    intersection = max(0, min(span_a[1], span_b[1]) - max(span_a[0], span_b[0]))
    union = max(span_a[1], span_b[1]) - min(span_a[0], span_b[0])
    return intersection / union if union != 0 else 0


# output annotation summary for all docs of the form: annotation category, # annotations, % out of total
# output the data to a file with annotator name
def output_annotation_summary(ann_name, annotations):
    total_annotations = len(annotations)
    csvwriter = csv.writer(open(f"{ann_name}_annotation_summary.csv", "w"))
    csvwriter.writerow(["label", "# annotations", "% out of total"])
    annotation_counts = {}
    for a in annotations:
        if a[2] not in annotation_counts:
            annotation_counts[a[2]] = 0
        annotation_counts[a[2]] += 1
    # sort by counts
    annotation_counts = dict(sorted(annotation_counts.items(), key=lambda item: item[1], reverse=True))
    for k in annotation_counts:
        # write to file
        csvwriter.writerow([k, annotation_counts[k], annotation_counts[k] / total_annotations * 100])
    csvwriter.writerow(["Total", total_annotations, "100%"])


def check_label_agreement(label_a, label_b):
    return label_a == label_b


def count_tokens(text):
    tokens = text.split()
    return len(tokens)


def extract_annotations(cas):
    annotations = []
    for fs in cas.select_all():
        if hasattr(fs, "label") and hasattr(fs, "begin") and hasattr(fs, "end") and fs.begin >= 0 and fs.end >= 0:
            tokens_cnt = count_tokens(fs.get_covered_text())
            # remove trailing - from label
            label = fs.label
            if fs.label.endswith("-"):
                fs.label = fs.label[:-1]
            if fs.label.startswith("-"):
                fs.label = fs.label[1:]
            annotations.append((fs.begin, fs.end, fs.label, tokens_cnt))
    # remove duplicates
    annotations = list(set(annotations))
    # sort annotation by start position treated as integers
    annotations = sorted(annotations, key=lambda x: int(x[0]))
    return annotations


# load annotations from all docs for two annotators

# inception_dir = os.environ['INCEPTION_FILES_DIR']
inception_dir = "./data-backup/files-1706796877"
if not os.path.isdir(inception_dir):
    print("Error: INCEPTION_FILES_DIR is not a directory")
    exit(1)
print("loading INCEpTION directory, path: " + inception_dir)

# load typesystem file
xml_file = os.path.join("..", "TypeSystem.xml")
if not os.path.isfile(xml_file):
    print("Error: TypeSystem.xml file not found in INCEpTION directory")
    exit(1)
print("loading typesystem file: " + xml_file)
with open(xml_file, 'rb') as f:
    typesystem = load_typesystem(f)

files_cnt = 0
empty_annotation_cnt = 0
total_annotation_instances = 0
multiple_annotators_cnt = 0
single_annotation_cnt = {}
two_annotation_cnt = 0
invalid_cnt = 0
agreement_cnt = 0
partial_agreement_cnt = 0
annotations_tokens_sum = 0
annotations_tokens_cnt = 0

docs = {}
annotations_of_annotator = {}
annotations_of_annotator["CURATION_USER"] = []
curated_docs = {}
# go over files and organize by document and annotator
for f in os.listdir(inception_dir):
    if f.endswith(".xmi"):
        xmi_file = os.path.join(inception_dir, f)
        # print("loading XMI file: " + xmi_file)
        with open(xmi_file, 'rb') as f:
            cas = load_cas_from_xmi(f, typesystem=typesystem)
        files_cnt += 1
        # extract document title and annotator username from document metadata
        md = cas.select("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")
        doc_id = md[0].documentTitle
        annotator = md[0].documentId

        annotations = extract_annotations(cas)
        if len(annotations) == 0:
            empty_annotation_cnt += 1
            if DEBUG:
                print(f"Error: no annotations for document {doc_id} and annotator {annotator}, skipping ...")
            continue  # do not include in stats

        if annotator == "naama":
            invalid_cnt += 1
            continue

        if annotator == "CURATION_USER":
            curated_docs[doc_id] = cas
            continue

        if doc_id not in docs:
            docs[doc_id] = {}
        if annotator not in docs[doc_id]:
            docs[doc_id][annotator] = []
        docs[doc_id][annotator].append(cas)

        total_annotation_instances += 1  # without curated docs, Naama and empty annotations

        for a in annotations:
            annotations_tokens_sum += a[3]
            annotations_tokens_cnt += 1

all_labels_ab_a = []
all_labels_ab_b = []
all_labels_ac_a = []
all_labels_bc_b = []
all_labels_ac_c = []
all_labels_bc_c = []

individual_agreement_results = {}
for doc_id in docs:
    if DEBUG:
        print("calculating agreement for document: " + doc_id)
    # if more than two annotators, skip document
    if len(docs[doc_id]) > 2:  # should never happen
        print(f"More than two annotators for document {doc_id}, annotators are {docs[doc_id].keys()} skipping ...")
        multiple_annotators_cnt += 1
        continue
    # if only one annotator, skip document
    if len(docs[doc_id]) < 2:
        if DEBUG:
            print(f"One annotator for document {doc_id}, skipping ...")
        annotator = list(docs[doc_id].keys())[0]
        if annotator not in single_annotation_cnt:
            single_annotation_cnt[annotator] = 0
        single_annotation_cnt[annotator] += 1
        continue
    two_annotation_cnt += 1
    # only if two annotators, calculate agreement
    annotators = list(docs[doc_id].keys())

    if (len(docs[doc_id][annotators[0]]) != 1) or (len(docs[doc_id][annotators[1]]) != 1):
        print(f"Error: more than one CAS for document {doc_id}, skipping ...")
        continue  # should never happen

    cas_a = docs[doc_id][annotators[0]][0]
    cas_b = docs[doc_id][annotators[1]][0]
    annotations_a = extract_annotations(cas_a)
    annotations_b = extract_annotations(cas_b)
    if DEBUG:
        print("the two annotators are " + annotators[0] + " and " + annotators[1])
    # if we have two annotators, store annotations for each annotator for stats
    if annotators[0] not in annotations_of_annotator:
        annotations_of_annotator[annotators[0]] = []
    annotations_of_annotator[annotators[0]].append(annotations_a)

    if annotators[1] not in annotations_of_annotator:
        annotations_of_annotator[annotators[1]] = []
    annotations_of_annotator[annotators[1]].append(annotations_b)

    labels_a, labels_b = extract_labels_partial_match(annotations_a, annotations_b, all_labels_ab_a, all_labels_ab_b)
    # calc agreement just for a single doc
    if len(set(labels_a).union(labels_b)) == 1:
        kappa_agreement_score = 1
    else:
        kappa_agreement_score = cohen_kappa_score(labels_a, labels_b)
    f1_agreement_score = f1_score(labels_a, labels_b, average='weighted')
    individual_agreement_results[doc_id] = (
    kappa_agreement_score, f1_agreement_score, len(annotations_a), len(annotations_b), annotations_a, annotations_b,
    labels_a, labels_b)

    # also calculate agreement of each annotator with the curated doc
    if doc_id in curated_docs.keys():
        curated_doc = curated_docs[doc_id]
        annotations_curated = extract_annotations(curated_doc)
        annotations_of_annotator["CURATION_USER"].append(annotations_curated) # for stats
        extract_labels_partial_match(annotations_a, annotations_curated, all_labels_ac_a, all_labels_ac_c)
        extract_labels_partial_match(annotations_b, annotations_curated, all_labels_bc_b, all_labels_bc_c)

# calc agreement for all docs
kappa_agreement_score_ab = cohen_kappa_score(all_labels_ab_a, all_labels_ab_b)
f1_agreement_score_ab = f1_score(all_labels_ab_a, all_labels_ab_b, average='weighted')

confusion_matrix_ab = confusion_matrix(all_labels_ab_a, all_labels_ab_b, labels=list(set(all_labels_ab_a).union(all_labels_ab_b)))
display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_ab, display_labels=list(set(all_labels_ab_a).union(all_labels_ab_b)))
display.plot(cmap=plt.cm.Blues)

# calc kappa for each annotator with the curated doc)
kappa_agreement_score_ac = cohen_kappa_score(all_labels_ac_a, all_labels_ac_c)
kappa_agreement_score_bc = cohen_kappa_score(all_labels_bc_b, all_labels_bc_c)

# calc f1 for each annotator with the curated doc
f1_agreement_score_ac = f1_score(all_labels_ac_a, all_labels_ac_c, average='weighted')
f1_agreement_score_bc = f1_score(all_labels_bc_b, all_labels_bc_c, average='weighted')

# calculate annotation counts for each annotator. Each summary is of the form: annotation category, # annotations, % out of total
for annotator in annotations_of_annotator:
    annotations = annotations_of_annotator[annotator]
    print(f"number of unique docs that were annotated by {annotator} annotator: {str(len(annotations))}")
    all_annotations = [item for sublist in annotations for item in sublist]
    output_annotation_summary(annotator, all_annotations)
    # print average number of annotations per document for each annotator
    avg_annotations = len(all_annotations) / len(annotations)
    print(f"Average number of annotations per document for annotator {annotator}: {avg_annotations}")

print("Invalid annotator docs count: " + str(invalid_cnt))
print("Multiple annotators unique docs count: " + str(multiple_annotators_cnt))
print("Single annotator unique docs count: " + str(single_annotation_cnt))
print("Two annotators unique docs count: " + str(two_annotation_cnt))

# count curated_docs docs
print("Curated docs count: " + str(len(curated_docs)))

# print average number of tokens per annotation
avg_tokens = annotations_tokens_sum / annotations_tokens_cnt
print("Average number of tokens per annotation: " + str(avg_tokens))
# number of documents that were annotated, if a document was annotated by more than one annotator, it is counted once for each annotator
# does not include curated docs, includes docs that have at least one annotation
print("Total annotation instances count: " + str(total_annotation_instances))
# files with no annotations
print("Empty annotation instances count: " + str(empty_annotation_cnt))
print("Total files count: " + str(files_cnt))
print(f"kappa: {str(kappa_agreement_score_ab)} ,iou_threshold: {iou_threshold}")
print(f"f1 agreement: {str(f1_agreement_score_ab)} ,iou_threshold: {iou_threshold}")
# print curation agreement
print(f"kappa for curation user with annotator a: {str(kappa_agreement_score_ac)} ,iou_threshold: {iou_threshold}")
print(f"kappa for curation user with annotator b: {str(kappa_agreement_score_bc)} ,iou_threshold: {iou_threshold}")

print(f"f1 agreement for curation user with annotator a: {str(f1_agreement_score_ac)} ,iou_threshold: {iou_threshold}")
print(f"f1 agreement for curation user with annotator b: {str(f1_agreement_score_bc)} ,iou_threshold: {iou_threshold}")


# save individual agreement results to csv file in the format: doc_id, kappa, f1, annotator1_annotations_count, annotator2_annotations_count
# sort by kappa
individual_agreement_results = dict(
    sorted(individual_agreement_results.items(), key=lambda item: item[1][0], reverse=True))
csvwriter = csv.writer(open("individual_agreement_results.csv", "w"))
csvwriter.writerow(
    ["doc_id", "kappa", "f1", "annotator1_annotations_count", "annotator2_annotations_count", "annotator1_annotations",
     "annotator2_annotations", "labels_a", "labels_b"])
for doc_id in individual_agreement_results:
    csvwriter.writerow([doc_id, individual_agreement_results[doc_id][0], individual_agreement_results[doc_id][1],
                        individual_agreement_results[doc_id][2], individual_agreement_results[doc_id][3],
                        individual_agreement_results[doc_id][4],
                        individual_agreement_results[doc_id][5], individual_agreement_results[doc_id][6],
                        individual_agreement_results[doc_id][7]])
print("done")

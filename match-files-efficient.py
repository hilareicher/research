import argparse
import os
import re
from difflib import SequenceMatcher
from glob import glob
from cassis import load_cas_from_xmi, load_typesystem
import Levenshtein


def group_texts_by_file(file_paths, text_loader):
    text_to_files = {}
    for path in file_paths:
        text = text_loader(path)
        if text in text_to_files:
            text_to_files[text].append(path)
        else:
            text_to_files[text] = [path]
    return text_to_files


def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def find_top_matches(texts_a, texts_b, method, top_n=3, limit=1000):
    top_matches = []
    for text_a in texts_a:
        if len(top_matches) >= limit:
            break
        # print progress every 2 files
        if len(top_matches) % 2 == 0:
            print(f"Processed {len(top_matches)} files")

        scores = [(index, get_similarity(text_a, text_b, method)) for index, text_b in enumerate(texts_b)]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_indices, top_scores = zip(*scores[:top_n]) if scores[:top_n] else ([], [])
        top_matches.append((top_indices, top_scores))
    return top_matches


def jaccard_similarity(a, b):
    a = set(a.split())
    b = set(b.split())
    return len(a & b) / len(a | b)


def levenshtein_similarity(s1, s2):
    return Levenshtein.ratio(s1, s2)


def get_similarity(a, b, method):
    if method == 'sequence':
        return SequenceMatcher(None, a, b).ratio()
    elif method == 'jaccard':
        return jaccard_similarity(a, b)
    elif method == 'levenshtein':
        return levenshtein_similarity(a, b)
    else:
        raise ValueError(f"Unknown similarity method: {method}")


def read_xmi(xmi_file, typesystem_p):
    with open(xmi_file, 'rb') as f:
        cas = load_cas_from_xmi(f, typesystem=typesystem_p)
        # extract document title and annotator username from document metadata
        md = cas.select("de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData")
        if not md:
            print(f"Error: DocumentMetaData not found in {xmi_file}")
            return None
        annotator = md[0].documentId
        if annotator == 'neta':
            content = cas.get_sofa().sofaString
            prefix_pattern = r"^.*?:\s{10,}"
            content_without_prefix = re.sub(prefix_pattern, "", content, count=1, flags=re.MULTILINE)
            return content_without_prefix
        else:
            return None


def main(directory_a, directory_b, method):
    xml_file = os.path.join(".", "TypeSystem.xml")
    if not os.path.isfile(xml_file):
        print("Error: TypeSystem.xml file not found in INCEpTION directory")
        exit(1)
    print("loading typesystem file: " + xml_file)
    with open(xml_file, 'rb') as f:
        typesystem = load_typesystem(f)

    file_paths_a = glob(f"{directory_a}/*.txt")
    texts_a = [read_text(path) for path in file_paths_a]
    print(f"Found {len(texts_a)} text files in {directory_a}")

    file_paths_b = glob(f"{directory_b}/*.xmi")
    # filter only files that were annotated by 'neta'
    texts_b = [read_xmi(path, typesystem) for path in file_paths_b]
    texts_b = [text for text in texts_b if text is not None]
    print(f"Found {len(texts_b)} XMI files in {directory_b}")

    print(f"Calculating similarity scores using {method} similarity ...")
    top_matches = find_top_matches(texts_b, texts_a, method, top_n=3)

    # Output matches
    csv_file = os.path.join(".", "reverse_matches.csv")
    with open(csv_file, 'w', encoding='utf-8') as file:
        file.write("file_xmi,matched_files,score\n")
        for b_index, (indices, scores) in enumerate(top_matches):
            for a_index, score in zip(indices, scores):
                base_filename_b = os.path.basename(file_paths_b[b_index])
                base_filename_a = os.path.basename(file_paths_a[a_index])
                file.write(f"{base_filename_b},{base_filename_a},{score}\n")

    for b_index, (indices, scores) in enumerate(top_matches):
        xmi_filename = os.path.basename(file_paths_b[b_index])
        print(f"XMI file {xmi_filename} matches with top 3 texts:")
        match_number = 1
        for match_index, score in zip(indices, scores):
            text_filename = os.path.basename(file_paths_a[match_index])
            # Print with match numbering
            print(f"  {match_number}. {text_filename} with similarity score {score}")
            match_number += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find top matching files')
    parser.add_argument('directory_a', type=str, help='Directory containing the text files')
    parser.add_argument('directory_b', type=str, help='Directory containing the XMI/XML files')
    parser.add_argument('--method', type=str, choices=['sequence', 'jaccard', 'levenshtein'], default='sequence',
                        help='Text similarity method to use: sequence, jaccard, or levenshtein')

    args = parser.parse_args()

    main(args.directory_a, args.directory_b, args.method)

# example usage:
# python match-files.py /path/to/txt/files /path/to/xmi/files --method sequence
# python match-files.py /path/to/txt/files /path/to/xmi/files --method jaccard
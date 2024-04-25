# convert xmi to txt by extracting the sofaString from the xmi file and writing it to a txt file
import argparse
import re

from cassis import load_typesystem, load_cas_from_xmi

# read directories from command line
parser = argparse.ArgumentParser(description='generate txt files from xmi files')
parser.add_argument('directory_a', type=str, help='directory for txt output files')
parser.add_argument('directory_b', type=str, help='directory with xmi input files')
args = parser.parse_args()

# list all xmi files in directory_b
import os
import glob

xmi_files = glob.glob(args.directory_b + '/*.xmi')

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



# load typesystem file
xml_file = os.path.join(args.directory_b, "TypeSystem.xml")
if not os.path.isfile(xml_file):
    print("Error: TypeSystem.xml file not found in INCEpTION directory")
    exit(1)
print("loading typesystem file: " + xml_file)
with open(xml_file, 'rb') as f:
    typesystem = load_typesystem(f)
texts_b = [read_xmi(path, typesystem) for path in xmi_files]
texts_b = [text for text in texts_b if text is not None]

print(f"Found {len(texts_b)} files in {xmi_files}")

# write texts to txt files
for i, text in enumerate(texts_b):
    with open(f"{args.directory_a}/{os.path.basename(xmi_files[i])}.txt", "w") as f:
        f.write(text)
print(f"Written {len(texts_b)} files to {args.directory_a}")

# Example usage:
# python generate-txt-from-xmi.py /path/to/txt/output /path/to/xmi/input


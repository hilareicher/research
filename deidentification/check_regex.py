import re

file_path = 'file_path'

removed_parts = []
prefix_pattern = r"^.*?:?\s{15,}"  # Match lines with a prefix of 15 or more spaces
before_double_return_pattern = r"^.*?\r?\n\r?\n"  # Match lines before double line breaks

# Read the content of the file
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
except FileNotFoundError:
    print(f"File not found: {file_path}")
    text = ""

# Search for the prefix pattern in the text
prefix_match = re.search(prefix_pattern, text, flags=re.MULTILINE)
if prefix_match:
    removed_parts.append(prefix_match.group())
    cleaned_text = re.sub(prefix_pattern, "", text, count=1, flags=re.MULTILINE)
    print("Found a match for the prefix pattern.")
else:
    cleaned_text = text
    print("No match found for the prefix pattern.")

# Search for the pattern before double returns in the cleaned text
double_return_match = re.search(before_double_return_pattern, cleaned_text, flags=re.DOTALL)
if double_return_match:
    removed_parts.append(double_return_match.group())
    cleaned_text = re.sub(before_double_return_pattern, "", cleaned_text, count=1, flags=re.DOTALL)
    print("Found a match for the double return pattern.")
else:
    print("No match found for the double return pattern.")

# Print the results
print("text:", cleaned_text)
print("***************************************************************")
print("title:", removed_parts[0] if removed_parts else "")

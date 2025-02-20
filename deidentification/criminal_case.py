import re

criminal_number_replacements = {}

def check_criminal_terms(original, entity_type):
    # List of Hebrew terms to check against
    hebrew_terms = ['בת"פ', 'מ"י', 'ת"פ', 'מ"ת']  # Add any additional terms here if necessary

    # Check if any term is in the original (for Hebrew terms)
    hebrew_match = any(term in original for term in hebrew_terms)

    # Regular expression pattern to match the case 33944-22-32 (digits separated by hyphens)
    # This will match any number of digits in the first part and two-digit numbers in the last two parts
    pattern = r'^\d{1,}-\d{2}-\d{2}$'

    # Check if original matches the pattern and if entity_type is not 'CRIMINAL_ID'
    pattern_match = bool(re.match(pattern, original)) and entity_type != 'CRIMINAL_ID'

    numeric_match = original.isdigit() and entity_type in ['ORG','LOC','COUNTRY']
    # Return True if either Hebrew term is matched or the pattern with entity_type is matched
    if hebrew_match or pattern_match or numeric_match :
        return True
    else:
        return False


def get_criminal_case(text):
    pattern = r"(?:ת\"פ|מ\"ת|מ\"י)\s*(\d{1,5}-\d{2}-\d{2})"
    matches = re.finditer(pattern, text)

    results = []
    for match in matches:
        start_position = match.start(1)  # Start of the captured group
        end_position = match.end(1)
        matched_text = match.group(1)  # The captured digits

        results.append({
            "text": matched_text,
            "maskOperator": "ranges",
            "textEntityType": "CRIMINAL_ID",
            # "textEntityType": "CRIMINAL_ID",
            "textStartPosition": start_position,
            "textEndPosition": end_position,
            "mask": ""
        })

    return results

def get_criminal_replacement(criminal_number):
    global criminal_number_replacements

    if criminal_number in criminal_number_replacements:
        return {"replacement_value": criminal_number_replacements[criminal_number],"justification": "Random digits"}
    return False


def add_criminal_replacement(original,replacement_digits):
    global criminal_number_replacements
    criminal_number_replacements[original]=replacement_digits



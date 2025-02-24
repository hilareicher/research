import re

import pandas as pd
from fuzzywuzzy import process

import utils

df = pd.read_csv('./datasets/meds_data.csv', encoding='utf-8')


def remove_prefix_and_identify(name, prefixes):
    """
    Remove specified prefixes from the beginning of a name and identify the prefix.

    Parameters:
        name (str): The name to process.
        prefixes (list): A list of prefixes to check and remove.

    Returns:
        tuple: A tuple containing the name without the prefix (if any) and the prefix itself.
               If no prefix is found, returns the original name and None.
    """
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):].strip(), prefix
    return name, None

def get_meds_original():
    # Create the regular expression
   return df['Original Name'].str.lower().tolist()

def get_meds_translated():
    # Create the regular expression
   return df['Translated Name'].tolist()


def check_meds_match(original, threshold=90):
    """
    Check if the given original city name matches any city in the DataFrame
    using fuzzy matching.

    Parameters:
        original (str): The city name to check.
        threshold (int): The score threshold for matching.

    Returns:
        bool: True if a match is found, False otherwise.
    """
    # original = clean_input(original)
    removed_prefixe = None
    meds_original = get_meds_original()
    meds_translated = get_meds_translated()

    # Combine both lists for matching
    all_meds = meds_original + meds_translated

    # Convert the input to lowercase for case-insensitive matching
    original_lower = original.lower()

    # Perform fuzzy matching
    best_match, score = process.extractOne(original_lower, all_meds)

    prefixes = ['ה', 'ב', 'ל', 'כ', 'מ', 'ו', 'ש', 'כש']
    modified_name, detected_prefix = remove_prefix_and_identify(original, prefixes)
    best_match_modified, score_modified = process.extractOne(modified_name, all_meds)

    # Debugging: Print the best match and score
    # print(f"Best match: {best_match}, Score: {score}")

    # Check if the score meets the threshold
    return score >= threshold or score_modified >= threshold
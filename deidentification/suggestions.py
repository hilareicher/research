# replacements.py
import string
import re
import city
import utils
from age import get_age_replacement
from city import get_city_replacement, get_random_city
from date import get_date_replacement
import orgs
from criminal_case import check_criminal_terms, get_criminal_replacement, add_criminal_replacement
import person
from orgs import get_org_replacement, is_org_in_mapping
from partial_date import get_partial_date_replacement
from person import get_name_replacement, get_random_name
from fuzzywuzzy import process

import re

id_map = {}
missing_parts_city = {}
SIGN = False


def suggest_replacement_for_date(original, patient_id, mask_operator):
    replacement_value = get_date_replacement(original, patient_id, mask_operator)
    # if date was replaced, then justification is 'Date shift'
    if replacement_value != original:
        return {"replacement_value": replacement_value, "justification": "Date shift"}
    return {"replacement_value": replacement_value, "justification": "Date/time format"}


# handle names (PER/PERS types)

def suggest_replacement_for_person(original, flag=False,prefixe=None, hospital=None):
    original = original.strip()
    # check if name appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    name_replacement = get_name_replacement(original)
    name_replacement["is_identifying_prefix"] = is_identifying_prefix
    if prefixe:
        name_replacement['replacement_value'] = prefixe+ name_replacement['replacement_value']
    # if name was not replaced and not because it was in the exclusion list then try to see if it's an organization
    if name_replacement["replacement_value"] == original and not name_replacement.get("in_exclusion_list", False):
        # check if it's in the organization list, if so then treat it as an organization
        if is_org_in_mapping(original):
            return suggest_replacement_for_org(original, hospital)
        return {"replacement_value": get_random_name(original), "unidentified_subtype": True,
                "is_identifying_prefix": is_identifying_prefix, "justification": "Name list"}
    return name_replacement


def suggest_replacement_for_org(original, flag=False,prefixe=None, hospital = None):
    print (f"org: {original}")
    original = original.strip()
    # print (f"suggesting replacement for org: {original}")
    # check if city appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    org_replacement = get_org_replacement(original, hospital)
    org_replacement["is_identifying_prefix"] = is_identifying_prefix
    if prefixe:
        org_replacement["replacement_value"] = prefixe + " " + org_replacement["replacement_value"]
    if org_replacement["replacement_value"] == original and not org_replacement.get("in_exclusion_list", False):
        # print (f"org {original} was not replaced and not in the exclusion list, checking if it's a city")
        # check if it's in the cities list, if so then treat it as a city
        if city.is_city(original):
            # print (f"org {original} is a city")
            return suggest_replacement_for_city(original, hospital)
        # print (f"org {original} was not found in the cities list, replacing with generic tag")
        return {"replacement_value": "ארגון", "unidentified_subtype": True,
                "is_identifying_prefix": is_identifying_prefix, "justification": "Mask"}
    return org_replacement


def suggest_replacement_for_city(original, hospital, flag=False,prefixe = None):
    original = original.strip()
    # check if city appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    city_replacement = get_city_replacement(original, hospital)
    if prefixe:
        city_replacement['replacement_value'] = prefixe+" "+city_replacement['replacement_value']
    city_replacement["is_identifying_prefix"] = is_identifying_prefix
    # if city was not replaced and not because it was in the exclusion list then try to see if it's an organization
    if city_replacement["replacement_value"] == original and not city_replacement.get("in_exclusion_list",
                                                                                      False) and not city_replacement.get(
            "above_population_threshold", False):
        # check if it's in the organization list, if so then treat it as an organization
        if is_org_in_mapping(original):
            return suggest_replacement_for_org(original, hospital)
        if flag:
            return {"replacement_value":prefixe + get_random_city(original), "unidentified_subtype": True,
                    "is_identifying_prefix": is_identifying_prefix, "justification": "City list"}
        return {"replacement_value":prefixe + get_random_city(original), "unidentified_subtype": True,
                "is_identifying_prefix": is_identifying_prefix, "justification": "City list"}
    if city_replacement.get("above_population_threshold", False):
        if flag:
            city_replacement["unidentified_subtype"] = "LARGE_CITY"
        city_replacement["unidentified_subtype"] = "LARGE_CITY"
    return city_replacement


def suggest_replacement_for_country(original, mask):
    return {"replacement_value": mask, "justification": "Country list"}


def suggest_replacement_for_age(original):
    return {"replacement_value": get_age_replacement(original), "justification": "Age range"}


def suggest_replacement_for_id(original):
    # Check if the original string is already in the id_map
    if original in id_map.keys():
        return {"replacement_value": id_map[original], "justification": "Random digits"}

    # Extract digits from the original string
    digits = ''.join(re.findall(r'\d', original))
    punctuated_digits = re.findall(r'\d+|[^\d]+', original)

    # Case 1: Six digits with no punctuation (ת.ז)
    if len(digits) == 6 and all(ch.isdigit() for ch in digits):
        replacement_value = ''.join(digits)  # No change, it's already valid
        return {"replacement_value": replacement_value, "justification": "ID (Six digits)"}

    # Case 2: Between 7 and 10 digits with punctuation after the first or second digit (ת.ז)
    if 7 <= len(digits) <= 10:
        # Check if there is punctuation in the string (e.g., hyphen, period, slash)
        if any(punc in original for punc in ['-', '.', '/']):
            # Find the index of the first punctuation character
            punc_index = min([original.find(punc) for punc in ['-', '.', '/'] if punc in original])

            # Check if the punctuation is after the first digit but not after the third
            if 1 <= punc_index < 3:  # Punctuation must be after the 1st or 2nd digit
                # Randomize all digits except the last 2
                replacement_digits = ''.join(utils.randomize_characters(digits))  # Randomize all except last 2
                # replacement_last_two = ''.join(utils.randomize_characters(digits[-2:]))  # Randomize the last 2 digits
                # replacement_value = replacement_digits + replacement_last_two

                # Reinsert the punctuation at its original index
                replacement_value = replacement_digits[:punc_index] + original[punc_index] + replacement_digits[
                                                                                             punc_index:]

                # Record in id_map
                id_map[original] = replacement_value
                return {"replacement_value": replacement_value,
                        "justification": "ID (7-10 digits with punctuation after 1st/2nd digit)"}
    #
    # Case 3: All other cases (not ת.ז)
    # We only modify the last two digits if we detect them to be problematic
    replacement_value = original
    if len(digits) > 2:
        replacement_digits = list(utils.randomize_characters(digits[-2:]))  # Only randomize last 2 digits
        replacement_value = original[:-2] + ''.join(replacement_digits)  # Append the random last 2 digits

    return {"replacement_value": replacement_value, "justification": "Not ID (Other cases)"}


# def suggest_replacement_for_id(original):
#     if original in id_map.keys():
#         return {"replacement_value": id_map[original], "justification": "Random digits"}
#
#     # Extract digits from the original string
#     digits = ''.join(re.findall(r'\d', original))
#
#     # Check if the number of digits is between 8 and 11
#     if 8 <= len(digits) <= 11:
#         if original in utils.exclusion_list:
#             return {
#                 "replacement_value": original,
#                 "in_exclusion_list": True,
#                 "justification": "Exclusion"
#             }
#         # Replace the digits with random characters, keeping non-digits unchanged
#         replacement_digits = list(utils.randomize_characters(digits))
#         replacement_value = re.sub(r'\d', lambda x: replacement_digits.pop(0), original, count=len(digits))
#         id_map[original] = replacement_value
#         return {"replacement_value": replacement_value, "justification": "Random digits"}
#
#     # If there are fewer than 8 or more than 11 digits
#     return {"replacement_value": original, "justification": "Invalid length or format"}

def suggest_replacement_for_criminal_cases(original):
    ans = get_criminal_replacement(original)
    if ans:
        return ans

    # Extract digits from the original string
    digits = ''.join(re.findall(r'\d', original))

    # Check if the number of digits is between 8 and 11
    if original in utils.exclusion_list:
        return {
            "replacement_value": original,
            "in_exclusion_list": True,
            "justification": "Exclusion",
        }
    # Replace the digits with random characters, keeping non-digits unchanged
    # replacement_digits = utils.randomize_characters(digits)
    replacement_digits = list(utils.randomize_characters(digits))
    replacement_value = re.sub(r'\d', lambda x: replacement_digits.pop(0), original)
    add_criminal_replacement(original, replacement_value)

    return {"replacement_value": replacement_value, "justification": "Random digits"}


def suggest_replacement_for_partial_date(original, patient_id, cleaned_date):
    replacement_value = get_partial_date_replacement(original, patient_id, cleaned_date)
    # if partial date was replaced, then justification is 'Date shift'
    if replacement_value != cleaned_date:
        return {"replacement_value": replacement_value, "justification": "Date shift"}
    return {"replacement_value": replacement_value, "justification": "Date/time format"}


def check_city_match(original, hospital, threshold=90):
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
    text_clean, removed = clean_input(original)
    if len(removed) > 0 and removed[0]:
        removed_prefixe = removed[0]
    cities = city.get_cities()
    best_match, score = process.extractOne(original, cities)
    if best_match == utils.HOSPITAL_HEBREW_MAP[hospital]:
        print("debug111")
        return True, best_match , removed_prefixe
    # Print the best match and score for debugging
    # print(f"Best match: {best_match}, Score: {score} , Original {original} :" )

    # Check if the score meets the threshold
    if score >= threshold:
        return True, best_match ,removed_prefixe

    return False, None ,removed_prefixe



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

def check_name_match(original, threshold=95):
    """
    Check if the given original city name matches any city in the DataFrame
    using fuzzy matching. Checks both with and without specific prefixes.

    Parameters:
        original (str): The city name to check.
        person (object): An object with a `get_name` method returning a list of names.
        threshold (int): The score threshold for matching.

    Returns:
        bool: True if a match is found, False otherwise.
        str: The best matching name, if found.
    """
    original = original.replace('"', '')

    # Get the list of names from the person object
    names = person.get_name()

    # Define prefixes to remove for additional matching
    prefixes = ['ה', 'ב', 'ל', 'כ', 'מ', 'ו', 'ש', 'כש']
    modified_name, detected_prefix = remove_prefix_and_identify(original, prefixes)

    # Perform fuzzy matching for both the original and modified names
    best_match_original, score_original = process.extractOne(original, names)
    best_match_modified, score_modified = process.extractOne(modified_name, names)

    # Select the best match based on the highest score
    if score_original >= threshold or score_modified >= threshold:
        if score_original >= score_modified:
            best_match = best_match_original
            score = score_original
            prefix = None
        else:
            best_match = best_match_modified
            score = score_modified
            prefix = detected_prefix

        # Clean the match if it contains additional data (e.g., slashes)
        if '/' in best_match:
            best_match = best_match.split('/')[0].strip()

        return True, best_match, prefix

    return False, None, None


# def check_name_match(original, threshold=90):
#     """
#     Check if the given original city name matches any city in the DataFrame
#     using fuzzy matching.
#
#     Parameters:
#         original (str): The city name to check.
#         threshold (int): The score threshold for matching.
#
#     Returns:
#         bool: True if a match is found, False otherwise.
#     """
#     original = original.replace('"', '')
#     names = person.get_name()
#     best_match, score = process.extractOne(original, names)
#
#     # Print the best match and score for debugging
#     # print(f"Best match: {best_match}, Score: {score}")
#
#     # Check if the score meets the threshold
#     if score >= threshold:
#         if '/' in best_match:
#             best_match = best_match.split('/')[0].strip()  # Take the first part and strip any spaces
#
#         return True, best_match
#
#     return False, None


# List of prefixes to remove
prefixes = [
    "מרג", "מרכז לבריאות הנפש", "המרכז הרפואי", "ביח", "ביהח",
    "בית חולים", "מרכז רפואי", "מרכז לבריאות הנפש", "בבית החולים","בית החולים",
    "בבית חולים",
    "בבית משפט" ,
    "בית משפט",
    "בית משפט השלום",
    "בבית משפט השלום",
    "במיון","מיון","ממיון",
        "בימ\"ש",
        "לבימ\"ש"
]



def clean_input(text):
    """
    Clean input text by removing specific unwanted prefixes and characters.
    Also returns the removed prefixes for tracking purposes.

    Parameters:
        text (str): The text to clean.

    Returns:
        tuple: Cleaned text and removed prefixes.
    """
    removed = []  # List to keep track of removed prefixes

    # Step 1: Remove unwanted prefixes
    prefix_pattern = r'\b(?:' + '|'.join(
        sorted([re.escape(prefix) for prefix in prefixes], key=len, reverse=True)) + r')\b'

    def remove_prefixes(match):
        removed.append(match.group().strip())  # Add matched prefix to the removed list
        return ''

    # Remove prefixes while tracking what was removed
    text = re.sub(prefix_pattern, remove_prefixes, text, flags=re.IGNORECASE)

    # Step 2: Remove all types of quotes (single, double, or multiple)
    text = re.sub(r'[\'"]+', '', text)

    # Step 3: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)

    # Step 4: Trim leading/trailing spaces
    cleaned_text = text.strip()

    return cleaned_text, removed

def check_org_match(original, hospital, threshold=90):
    """
    Check if the given original city name matches any city in the DataFrame
    using fuzzy matching.

    Parameters:
        original (str): The city name to check.
        threshold (int): The score threshold for matching.

    Returns:
        bool: True if a match is found, False otherwise.
    """
    # original = original.replace('"', '')
    removed_prefixe = None
    text_clean,removed  = clean_input(original)
    if len(removed) > 0 and removed[0] :
        removed_prefixe = removed[0]
    # if text_clean != original:
    #     pass
    cities = orgs.get_orgs_split()
    best_match, score = process.extractOne(text_clean, cities)
    if best_match == utils.HOSPITAL_HEBREW_MAP[hospital]:
        return True,best_match,removed_prefixe
    # Print the best match and score for debugging
    # print(f"Best match: {best_match}, Score: {score}")

    # Check if the score meets the threshold
    if score >= threshold:
        if '/' in best_match:
            best_match = best_match.split('/')[0].strip()  # Take the first part and strip any spaces

        return True, best_match,removed_prefixe

    return False, None ,removed_prefixe


def handle_city_parts(original, hospital):
    global SIGN
    """
    Handle cases where a city name arrives in parts (e.g., 'Gan-' and then 'Yavne').

    Parameters:
        original (str): The original city name or part of the city name.

    Returns:
        str: The full city name if a match is found, or the partial city name.
    """
    # Check if the city matches any in the database
    score_flag, city_match = check_city_match(original, hospital)

    if score_flag:
        # Handle the case where the city name has a hyphen ('-') indicating it's incomplete
        if '-' in original:
            SIGN = True
            if city_match not in missing_parts_city:
                trimmed_original = original.strip()  # Remove spaces from both ends of original
                missing_part = city_match.replace(original.replace("-", "").strip(),
                                                  "").strip()  # Remove spaces from city match
                missing_parts_city[city_match] = [trimmed_original, missing_part]
                # print(f"Added '{original}' to missing parts dictionary with count 1.")
        else:
            for partial_city, parts in missing_parts_city.items():
                if original in parts:
                    SIGN = False
                return True

    return False


def suggest_replacement_for_url(original_url):
    # Define a pattern for common URL components
    # url_pattern = r'(http://|https://|www\.|ftp://|ftps://|@|[/?&=:\-\.])'
    url_pattern = r'^(http://|https://|ftp://|ftps://|www\.|[^@\s]+@[^@\s]+\.[^@\s]+)'
    # Check if the URL contains valid URL components
    if original_url in utils.exclusion_list:
        return {
            "replacement_value": original_url,
            "in_exclusion_list": True,
            ""
            "justification": "Exclusion"
        }

    if re.search(url_pattern, original_url):
        # Replace the digits with random characters, keeping non-digits unchanged
        replacement_digits = utils.randomize_characters(original_url)
        # replacement_value = re.sub(r'\d', lambda x: replacement_digits[0], original_url, count=len(original_url))
        return {"replacement_value": replacement_digits, "justification": "Random digits"}

    # If the URL does not contain valid components
    return {
        "replacement_value": original_url,
        "justification": "Invalid URL format"
    }


def suggest_replacement(entity_type, original, doc_id, mask_operator, mask, hospital):
    global SIGN
    parts = doc_id.split('_')  # e.g., 34341_01_13432_111.txt
    patient_id = parts[0]
    if check_criminal_terms(original, entity_type):
        return False
    if entity_type == 'DATE' or entity_type == 'DATE_TIME' or entity_type == 'LATIN_DATE' or entity_type == 'NOISY_DATE' or entity_type == 'PREPOSITION_DATE':
        return suggest_replacement_for_date(original, patient_id, mask_operator)

    if entity_type == 'PERS' or entity_type == 'PER':
        score_flag, org_match,prefix = check_name_match(original)
        if score_flag:
            return suggest_replacement_for_person(org_match, True,prefix, hospital)
        # todo - change1  suggest_replacement_for_person(org_match, True) -> return {"replacement_value": original}

        return {"replacement_value": original}
        # return suggest_replacement_for_person(original, True)
    elif entity_type == 'ORG' or entity_type == 'FAC' or entity_type == 'GPE':
        score_flag, org_match,removed_prefixe = check_org_match(original, hospital, threshold=90)
        if score_flag:
            return suggest_replacement_for_org(org_match, True, removed_prefixe, hospital)

        score_flag, city_match,removed_prefixe = check_city_match(original, hospital, threshold=90)
        if score_flag:
            return suggest_replacement_for_city(city_match, hospital, True,removed_prefixe)
        return {"replacement_value": original}

    elif entity_type == 'LOC':
        score_flag, city_match,removed_prefixe = check_city_match(original, hospital)
        if score_flag:
            if '-' in original:
                _ = handle_city_parts(original, hospital)
            return suggest_replacement_for_city(city_match, hospital,True,removed_prefixe)

        score_flag, org_match,removed_prefixe = check_org_match(original, hospital)
        if score_flag:
            return suggest_replacement_for_org(org_match, True,removed_prefixe, hospital)

        return {"replacement_value": original}

        # return suggest_replacement_for_org(original,True)

    elif entity_type == 'CITY':
        if SIGN:
            flag = handle_city_parts(original, hospital)
            if flag:
                return {"replacement_value": " ", "unidentified_subtype": True}
        score_flag, city_match,removed_prefixe = check_city_match(original, hospital, threshold=95)
        if score_flag:
            return suggest_replacement_for_city(city_match, hospital,True)
            # return suggest_replacement_for_city(original)
        return {"replacement_value": original}
    elif entity_type == 'COUNTRY':
        return suggest_replacement_for_country(original, mask)
    elif entity_type == 'AGE':
        return suggest_replacement_for_age(original)
    # elif entity_type == 'ID' or entity_type == 'ISRAELI_ID_NUMBER' or entity_type == 'PHONE_NUMBER' or entity_type == "URL" or entity_type == "EMAIL_ADDRESS":

    elif entity_type == 'ID' or entity_type == 'ISRAELI_ID_NUMBER' or entity_type == 'PHONE_NUMBER':
        return suggest_replacement_for_id(original)
    elif entity_type == "URL" or entity_type == "EMAIL_ADDRESS":
        return suggest_replacement_for_url(original)
        # return suggest_replacement_for_url(original), None

    elif entity_type == 'MISC__AFF':
        return {"replacement_value": original, "justification": "MISC__AFF"}
    elif entity_type == 'TIME':
        return {"replacement_value": original, "justification": "TIME"}
    elif entity_type == 'CRIMINAL_ID':
        return suggest_replacement_for_criminal_cases(original)
    elif entity_type == 'PARTIAL_DATE':
        return suggest_replacement_for_partial_date(original, patient_id, mask)
    else:
        return {"replacement_value": original}

# replacements.py
import string

import city
import utils
from age import get_age_replacement
from city import get_city_replacement, get_random_city
from date import get_date_replacement
from orgs import get_org_replacement, is_org_in_mapping
from partial_date import get_partial_date_replacement
from person import get_name_replacement, get_random_name


def suggest_replacement_for_date(original, patient_id, mask_operator):
    replacement_value = get_date_replacement(original, patient_id, mask_operator)
    # if date was replaced, then justification is 'Date shift'
    if replacement_value != original:
        return {"replacement_value": replacement_value, "justification":"Date shift"}
    return {"replacement_value":replacement_value, "justification":"Date/time format"}


# handle names (PER/PERS types)

def suggest_replacement_for_person(original):
    original = original.strip()
    # check if name appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    name_replacement = get_name_replacement(original)
    name_replacement["is_identifying_prefix"] = is_identifying_prefix
    # if name was not replaced and not because it was in the exclusion list then try to see if it's an organization
    if name_replacement["replacement_value"] == original and not name_replacement.get("in_exclusion_list",False):
        # check if it's in the organization list, if so then treat it as an organization
        if is_org_in_mapping(original):
            return suggest_replacement_for_org(original)
        return {"replacement_value": get_random_name(original) , "unidentified_subtype": True, "is_identifying_prefix": is_identifying_prefix, "justification":"Name list"}
    return name_replacement


def suggest_replacement_for_org(original):
    original = original.strip()
    #print (f"suggesting replacement for org: {original}")
    # check if city appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    org_replacement = get_org_replacement(original)
    org_replacement["is_identifying_prefix"] = is_identifying_prefix
    if org_replacement["replacement_value"] == original and not org_replacement.get("in_exclusion_list",False):
        #print (f"org {original} was not replaced and not in the exclusion list, checking if it's a city")
        # check if it's in the cities list, if so then treat it as a city
        if city.is_city(original):
            #print (f"org {original} is a city")
            return suggest_replacement_for_city(original)
        #print (f"org {original} was not found in the cities list, replacing with generic tag")
        return {"replacement_value": "ארגון", "unidentified_subtype": True, "is_identifying_prefix": is_identifying_prefix, "justification":"Mask"}
    return org_replacement


def suggest_replacement_for_city(original):
    original = original.strip()
    # check if city appears in the identifying prefix list, if it does, just mark it with a special indicator
    is_identifying_prefix = original in utils.identifying_prefixes
    city_replacement = get_city_replacement(original)
    city_replacement["is_identifying_prefix"] = is_identifying_prefix
    # if city was not replaced and not because it was in the exclusion list then try to see if it's an organization
    if city_replacement["replacement_value"] == original and not city_replacement.get("in_exclusion_list",False) and not city_replacement.get("above_population_threshold",False):
        # check if it's in the organization list, if so then treat it as an organization
        if is_org_in_mapping(original):
            return suggest_replacement_for_org(original)
        return {"replacement_value": get_random_city(original) , "unidentified_subtype": True, "is_identifying_prefix": is_identifying_prefix, "justification":"City list"}
    if city_replacement.get("above_population_threshold", False):
        city_replacement["unidentified_subtype"] = "LARGE_CITY"
    return city_replacement


def suggest_replacement_for_country(original, mask):
    return {"replacement_value": mask, "justification": "Country list"}


def suggest_replacement_for_age(original):
    return {"replacement_value": get_age_replacement(original), "justification": "Age range"}


def suggest_replacement_for_id(original):
    if original in utils.exclusion_list:
        return {
            "replacement_value": original,
            "in_exclusion_list": True,
            "justification": "Exclusion"
        }
    replacement_value = utils.randomize_characters(original)
    return {"replacement_value": replacement_value, "justification": "Random digits"}


def suggest_replacement_for_partial_date(original, patient_id, cleaned_date):
    replacement_value = get_partial_date_replacement(original, patient_id, cleaned_date)
    # if partial date was replaced, then justification is 'Date shift'
    if replacement_value != cleaned_date:
        return {"replacement_value": replacement_value, "justification":"Date shift"}
    return {"replacement_value":replacement_value, "justification":"Date/time format"}


def suggest_replacement(entity_type, original, doc_id, mask_operator, mask):
    parts = doc_id.split('_')  # e.g., 34341_01_13432_111.txt
    patient_id = parts[0]

    if entity_type == 'DATE' or entity_type == 'DATE_TIME' or entity_type == 'LATIN_DATE' or entity_type == 'NOISY_DATE' or entity_type == 'PREPOSITION_DATE':
        return suggest_replacement_for_date(original, patient_id, mask_operator)
    elif entity_type == 'PERS' or entity_type == 'PER':
        return suggest_replacement_for_person(original)
    elif entity_type == 'ORG' or entity_type == 'FAC' or entity_type == 'GPE' or entity_type == 'LOC':
        return suggest_replacement_for_org(original)
    elif entity_type == 'CITY':
        return suggest_replacement_for_city(original)
    elif entity_type == 'COUNTRY':
        return suggest_replacement_for_country(original, mask)
    elif entity_type == 'AGE':
        return suggest_replacement_for_age(original)
    elif entity_type == 'ID' or entity_type == 'ISRAELI_ID_NUMBER' or entity_type == 'PHONE_NUMBER' or entity_type == "URL" or entity_type == "EMAIL_ADDRESS":
        return suggest_replacement_for_id(original)
    elif entity_type == 'MISC__AFF':
        return {"replacement_value": original, "justification": "MISC__AFF"}
    elif entity_type == 'TIME':
        return {"replacement_value": original, "justification": "TIME"}
    elif entity_type == 'PARTIAL_DATE':
        return suggest_replacement_for_partial_date(original, patient_id, mask)
    else:
        return {"replacement_value": original}

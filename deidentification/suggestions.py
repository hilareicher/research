# replacements.py
import string

from age import get_age_replacement
from city import get_city_replacement
from date import get_date_replacement
from location import get_location_replacement
from org import get_org_replacement
from person import get_name_replacement


def suggest_replacement_for_date(original, doc_id, mask_operator):
    parts = doc_id.split('_')  # e.g., 34341_01_13432_111.txt
    patient_id = parts[0]
    return get_date_replacement(original, patient_id, mask_operator)


# handle names (PER/PERS types)
def suggest_replacement_for_person(original):
    return " ".join([get_name_replacement(word) for word in original.split()])


def suggest_replacement_for_org(original):
    return get_org_replacement(original)


def get_loc_replacement(original):
    return get_location_replacement(original)


def suggest_replacement_for_loc(original):
    return get_loc_replacement(original)


def suggest_replacement_for_city(original):
    return get_city_replacement(original)


def suggest_replacement_for_country(original, mask):
    return mask


def suggest_replacement_for_age(original):
    return get_age_replacement(original)


def suggest_replacement_for_id(original):
    return "12345678"


def suggest_replacement(entity_type, original, doc_id, mask_operator, mask):
    if entity_type == 'DATE' or entity_type == 'DATE_TIME':
        return suggest_replacement_for_date(original, doc_id, mask_operator)
    elif entity_type == 'PERS' or entity_type == 'PER':
        return suggest_replacement_for_person(original)
    elif entity_type == 'ORG':
        return suggest_replacement_for_org(original)
    elif entity_type == 'LOC':
        return suggest_replacement_for_loc(original)
    elif entity_type == 'CITY':
        return suggest_replacement_for_city(original)
    elif entity_type == 'COUNTRY':
        return suggest_replacement_for_country(original, mask)
    elif entity_type == 'AGE':
        return suggest_replacement_for_age(original)
    elif entity_type == 'ID' or entity_type == 'ISRAELI_ID_NUMBER' or entity_type == 'PHONE_NUMBER':
        return suggest_replacement_for_id(original)
    else:
        return original

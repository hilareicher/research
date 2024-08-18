import utils
from utils import starts_with_formative_letters
import pandas as pd
import random

lev_hasharon = 'לב השרון'
lev_hasharon_replacement = 'מוסדנו'

org_df = pd.DataFrame()
org_replacements = {}
org_mapping = {}


prefixes = ["מרג", "מרכז לבריאות הנפש", "המרכז הרפואי", "ביח", "ביהח", "בית חולים", "מרכז רפואי", "מרכז לבריאות הנפש", "בית החולים"]

def load_org_data(filename='./datasets/orgs.csv'):
    global org_df, org_mapping
    org_df = pd.read_csv(filename).sort_values(by=['type', 'org'], ascending=[True, True])
    for _, row in org_df.iterrows():
        orgs = row['org'].split('/')
        main_org = orgs[0].strip()
        org_type = row['type'].strip()
        for org in orgs:
            org_mapping[org.strip()] = (main_org, org_type)


def get_org_replacement(org_name):
    global org_mapping, org_replacements

    if org_name in utils.exclusion_list:
        return {
            "replacement_value": org_name,
            "in_exclusion_list": True,
            "justification": "Exclusion"
        }

    if lev_hasharon in org_name:
        return {"replacement_value": org_name.replace(lev_hasharon, lev_hasharon_replacement), "justification": "Mask"}

    adjusted_org_name, removed_formatives_and_prefix = adjust_org_if_needed(org_name)

    if adjusted_org_name is None:
        print(f"name {org_name} not found in names dataset")
        return {
            "replacement_value": org_name,
        }

    main_org, org_type = org_mapping[adjusted_org_name]
    if main_org in org_replacements:
        # attach the removed formatives to the replacement
        replacement_org = removed_formatives_and_prefix + org_replacements[main_org] if removed_formatives_and_prefix else org_replacements[main_org]
        return {"replacement_value": replacement_org, "justification":"Organization list"}

    potential_replacements = org_df[org_df['type'] == org_type]['org'].tolist()
    potential_replacements = [name.split('/')[0].strip() for name in potential_replacements]
    potential_replacements = list(set(potential_replacements) - set(org_replacements.values()))

    if not potential_replacements:
        return {"replacement_value": org_name}

    replacement_org = random.choice(potential_replacements)
    org_replacements[main_org] = replacement_org
    replacement_org = removed_formatives_and_prefix + replacement_org if removed_formatives_and_prefix else replacement_org

    return {"replacement_value": replacement_org, "justification":"Organization list"}


def is_org_in_mapping(org_name):
    org_name = org_name.strip()

    if org_name in org_mapping:
        return True

    adjusted_org_name, _ = adjust_org_if_needed(org_name)
    return adjusted_org_name is not None


def adjust_org_if_needed(org_name):
    if org_name in org_mapping:
        return org_name, ""

    org_name = org_name.replace('"', '')
    org_name = org_name.replace('-', '')
    org_name = org_name.replace('״', '')

    if org_name in org_mapping:
        return org_name, ""


    original_org_name = org_name
    removed_formatives_and_prefix = ""

    # Check for one formative letter
    if starts_with_formative_letters(org_name):
        removed_formatives_and_prefix = org_name[0]
        org_name = org_name[1:]
        if org_name in org_mapping:
            return org_name, removed_formatives_and_prefix

    # Check for prefix without formative letter
    org_name = original_org_name
    removed_formatives_and_prefix = ""
    for prefix in prefixes:
        if org_name.startswith(prefix):

            removed_formatives_and_prefix = prefix + " "
            org_name = org_name[len(prefix):].strip()
            if org_name in org_mapping:
                return org_name, removed_formatives_and_prefix


    # Check for prefix with one formative letter
    if starts_with_formative_letters(original_org_name):
        first_letter = original_org_name[0]
        org_name = original_org_name[1:]
        for prefix in prefixes:
            if org_name.startswith(prefix):
                removed_formatives_and_prefix = first_letter + prefix + " "
                org_name = org_name[len(prefix):].strip()
                if org_name in org_mapping:
                    return org_name, removed_formatives_and_prefix

    # Check for two formative letters
    org_name = original_org_name
    removed_formatives_and_prefix = ""
    for _ in range(2):
        if starts_with_formative_letters(org_name):
            removed_formatives_and_prefix += org_name[0]
            org_name = org_name[1:]
        else:
            break
        if org_name in org_mapping:
            return org_name, removed_formatives_and_prefix

    # Check for two formative letters with prefix
    if len(original_org_name) > 1 and starts_with_formative_letters(original_org_name[0]) and starts_with_formative_letters(original_org_name[1]):
        first_two_letters = original_org_name[:2]
        org_name = original_org_name[2:]
        for prefix in prefixes:
            if org_name.startswith(prefix):
                removed_formatives_and_prefix = first_two_letters + prefix + " "
                org_name = org_name[len(prefix):].strip()
                if org_name in org_mapping:
                    return org_name, removed_formatives_and_prefix

    return None, ""



# Load the organization data
load_org_data()

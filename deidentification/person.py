import pandas as pd
import random

import utils
from utils import starts_with_formative_letters, remove_punc

names_df = pd.DataFrame()
replacements = {}


def load_names_data(filename='./datasets/names.csv'):
    global names_df
    names_df = pd.read_csv(filename).sort_values(by=['Name', 'Count'], ascending=[True, False])
    names_df = names_df.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)


load_names_data()


def get_name_replacement(name):
    global names_df, replacements

    if name in utils.exclusion_list:
        return {
            "replacement_value": name,
            "in_exclusion_list": True,
            "justification": "Exclusion"
        }

    # Split the name and process each part
    parts = name.split()
    replacement_parts = []
    in_exclusion_list = False
    unidentified_subtype = False
    accumulated_justification = ""

    for part in parts:
        part_replacement = process_name_part(part)
        replacement_parts.append(part_replacement["replacement_value"])
        if part_replacement.get("in_exclusion_list", False):
            in_exclusion_list = True
        if part_replacement.get("unidentified_subtype", False):
            unidentified_subtype = True
        # append the justification to the accumulated justification with a comma if it's not the first justification
        if len(accumulated_justification) > 0:
            accumulated_justification += ", "
        accumulated_justification += part_replacement.get("justification", "NA")

    replacement_value = " ".join(replacement_parts)
    result = {
        "replacement_value": replacement_value,
        "in_exclusion_list": in_exclusion_list,
        "unidentified_subtype": unidentified_subtype,
        "justification": accumulated_justification
    }
    return result


def process_name_part(part):
    original_part = part
    global names_df, replacements

    part = part.strip()

    if part in utils.exclusion_list:
        return {
            "replacement_value": part,
            "in_exclusion_list": True,
            "justification": "Exclusion"
        }

    part = remove_punc(part)

    if part in replacements:
        return {
            "replacement_value": replacements[part],
            "justification": "Name list"
        }

    adjusted_name, formative_letters = adjust_name_if_needed(part)

    if adjusted_name is None:
        print(f"name {part} not found in names dataset")
        return {
            "replacement_value": original_part
        }

    if adjusted_name in replacements:
        replacement_value = formative_letters + replacements[adjusted_name] if formative_letters else replacements[
            adjusted_name]
        return {
            "replacement_value": replacement_value,
            "justification": "Name list"
        }

    name_type = names_df.loc[names_df['Name'] == adjusted_name, 'Type'].iloc[0]
    potential_replacements_df = names_df[
        (names_df['Type'] == name_type) &
        (~names_df['Name'].isin(replacements.values())) &
        (names_df['Name'] != adjusted_name)
        ]

    if potential_replacements_df.empty:
        print(f"No unique replacement found for {part}")
        return {
            "replacement_value": part,
            "unidentified_subtype": True
        }

    potential_replacements = potential_replacements_df['Name'].tolist()
    weights = potential_replacements_df['Count'].tolist()

    sampled_name = random.choices(potential_replacements, weights=weights, k=1)[0]
    replacements[adjusted_name] = sampled_name

    replacement_value = formative_letters + sampled_name if formative_letters else sampled_name
    return {
        "replacement_value": replacement_value,
        "justification": "Name list"
    }


def adjust_name_if_needed(name):
    if any(names_df['Name'] == name):
        return name, None

    removed_formatives = ""
    original_name = name

    for _ in range(2):
        if starts_with_formative_letters(name):
            removed_formatives = name[0] + removed_formatives
            name = name[1:]
            if any(names_df['Name'] == name):
                return name, removed_formatives
    return None, None


# Replacement suggestion function for persons
def suggest_replacement_for_person(original):
    return get_name_replacement(original)


def get_random_name(original):
    potential_replacements_df = names_df[~names_df['Name'].isin(replacements.values())]
    names_list = potential_replacements_df['Name'].tolist()
    weights_list = potential_replacements_df['Count'].tolist()
    sampled_name = random.choices(names_list, weights=weights_list, k=1)[0]
    replacements[original] = sampled_name
    return sampled_name

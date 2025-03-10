import re

import pandas as pd

import utils

replacement = 'מוסדנו'

df = pd.read_csv('./datasets/cities.csv', encoding='utf-8')

selected_cities = {}
# Convert population column to integer to use as weights
df['סה"כ'] = df['סה"כ'].apply(lambda x: int(x.replace(",", "")))



def get_cities_from_text(text, hospital):
    # Create the regular expression
    orgs = [org.strip() for sublist in df['שם_ישוב'].str.split('/') for org in sublist if org.strip()]
    pattern = r"\b(?:(?:ה|ל|מ|ב|ש)?)(" + "|".join(re.escape(place) for place in orgs) + r")\b"
    matches = re.finditer(pattern, text)
    results = []
    expected_hospital_hebrew = utils.HOSPITAL_HEBREW_MAP[hospital]

    # Iterate over the matches
    for match in matches:
        start_position = match.start(1)  # Start of the match
        end_position = match.end(1)  # End of the match
        matched_text = match.group(1)  # The matched text
        if matched_text == expected_hospital_hebrew:
            continue
        # Append the relevant data to the results
        results.append({
            "text": matched_text,
            "maskOperator": "ranges",
            "textEntityType": "CITY",
            # "textEntityType": "ORG",
            "textStartPosition": start_position,
            "textEndPosition": end_position,
            "mask": ""
        })

    return results

def get_selected_cities():
    return list(selected_cities.keys())

def get_cities():
    return df['שם_ישוב'].str.strip().tolist()

def get_city_replacement(city_name, hospital, population_threshold=50000000):
    # global selected_cities
    # print (f"city_name: {city_name}")
    if city_name in utils.exclusion_list:
        # print (f"city {city_name} is in the exclusion list")
        return {
            "replacement_value": city_name,
            "in_exclusion_list": True,
            "justification": "Exclusion"
        }

    hospital_hebrew = utils.HOSPITAL_HEBREW_MAP.get(hospital)
    if hospital_hebrew in city_name:
        return {
            "replacement_value": city_name.replace(hospital_hebrew, replacement),
            "justification": "Mask"
        }

    if city_name in ['תל-אביב', 'תל אביב', 'ת"א', 'תל אביב יפו', 'תל-אביב יפו']:
        return  {"replacement_value": city_name, "above_population_threshold":True, "justification":"Large City"} # all forms of Tel Aviv, should not be replaced

    adjusted_city_name, removed_formatives = adjust_city_if_needed(city_name)

    if adjusted_city_name in selected_cities:
        # print (f"city {city_name} already selected")
        return {"replacement_value": removed_formatives + selected_cities[adjusted_city_name] if removed_formatives else selected_cities[adjusted_city_name], "justification":"City list" }
    city_info = df[df['שם_ישוב'] == adjusted_city_name]

    if city_info.empty:
        # print(f"name {city_name} not found in cities dataset")
        return {
            "replacement_value": city_name,
            "unidentified_subtype": True
        }

    # if the city is found but does not meet the population threshold
    elif city_info['סה"כ'].values[0] <= population_threshold:
        # print (f"city {city_name} does not meet the population threshold")
        jewish_or_arab = city_info['מגזר'].values[0]
        location = city_info['פריפריה/ מרכז'].values[0]
        alternative_cities = df[(df['מגזר'] == jewish_or_arab) &
                                (df['פריפריה/ מרכז'] == location) &
                                (~df['שם_ישוב'].isin(selected_cities.values()))]

        # if there are alternatives, choose one at random
        if not alternative_cities.empty:
            alternative_city_name = alternative_cities.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
            selected_cities[adjusted_city_name] = alternative_city_name
            # print (f"selected alternative city: {alternative_city_name}")
            return {"replacement_value": removed_formatives + alternative_city_name if removed_formatives else alternative_city_name, "justification":"City list"}
        else:
            random_city_name = df.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
            selected_cities[adjusted_city_name] = random_city_name
            # print (f"selected random city: {random_city_name}")
            return {"replacement_value": random_city_name, "justification":"City list"}
    else:
        # print (f"for city {city_name} the population is above the threshold")
        return {"replacement_value": city_name, "above_population_threshold": True, "justification":"Large City"}


def get_random_city(original):
    global selected_cities

    alternative_cities = df[~df['שם_ישוב'].isin(selected_cities.values())]
    random_city_name = alternative_cities.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
    selected_cities[original] = random_city_name
    return random_city_name


def adjust_city_if_needed(original):
    if original in df['שם_ישוב'].values:
        return original, None

    removed_formatives = ""

    for _ in range(2):
        if utils.starts_with_formative_letters(original):
            removed_formatives = original[0] + removed_formatives
            original = original[1:]
            if original in df['שם_ישוב'].values:
                return original, removed_formatives
    return None, None

def is_city(original):
    original = original.strip()

    if original in df['שם_ישוב'].values:
        return True

    adjusted_city_name, _ = adjust_city_if_needed(original)
    return adjusted_city_name is not None
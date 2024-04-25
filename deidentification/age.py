import re


def get_age_replacement(age):
    age = int(age)
    if age < 18:
        return "מתחת ל-18"
    elif age > 100:
        return "מעל ל-100"

    if 18 <= age <= 25:
        return "18-25"

    # Calculate the upper and lower bounds for ages 25+
    lower_bound = (age - 1) // 5 * 5
    upper_bound = lower_bound + 5

    #  lower bound for the first range after 18-25
    if lower_bound == 25:
        lower_bound = 26

    return f"{lower_bound}-{upper_bound}"


def get_age_entities(text):
    pattern = r"(?:בן|בת)\s+(\d{1,2})"
    matches = re.finditer(pattern, text)

    results = []
    for match in matches:
        start_position = match.start(1)  # Start of the captured group
        matched_text = match.group(1)  # The captured digits

        results.append({
            "text": matched_text,
            "maskOperator": "ranges",
            "textEntityType": "AGE",
            "textStartPosition": start_position,
            "mask": ""
        })

    return results

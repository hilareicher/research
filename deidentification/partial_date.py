import re
from datetime import datetime, timedelta

from date import get_shift_for_patient


def shift_partial_date(date_str, shift_days):
    # parse partial date assuming it is in DD/MM format
    day, month = map(int, date_str.split('/'))

    # create datetime object - date is partial so assume current year
    date = datetime(datetime.now().year, month, day)

    # Shift the date by the specified number of days
    shifted_date = date + timedelta(days=shift_days)

    # Format the shifted date as DD/MM
    shifted_date_str = shifted_date.strftime("%d/%m")

    return shifted_date_str

def get_partial_date_replacement(original, patient_id, clean_date):
    shift_days = get_shift_for_patient(patient_id)
    return shift_partial_date(clean_date, shift_days)



def get_partial_date_entities(text):
    pattern = r'\b(0?[1-9]|[12][0-9]|3[01])[-/. ](0?[1-9]|1[0-2])\b(?![-/. ]\d{2,4})' # DD/MM, D/M, D/MM, DD/M, exclude dates with year part

    matches = re.finditer(pattern, text)

    results = []
    for match in matches:
        start_position = match.start()  # Start of the captured group
        end_position = match.end()
        matched_text = match.group()  # The captured digits

        # Constructing the cleaned date in DD/MM format
        day = match.group(1)
        month = match.group(2)
        cleaned_date = f"{day}/{month}"

        results.append({
            "text": matched_text,
            "maskOperator": "date_shift",
            "textEntityType": "PARTIAL_DATE",
            "textStartPosition": start_position,
            "textEndPosition": end_position,
            "mask": cleaned_date # we use the mask to store the cleaned date
        })

    return results

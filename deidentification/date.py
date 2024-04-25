from datetime import datetime, timedelta
import re
import hashlib

from date_utils import extract_date_components, extract_numerical_date_components


def hash_patient_id(patient_id):
    # use SHA-256 hash function and take the first few characters to convert to an integer
    hash_object = hashlib.sha256(patient_id.encode())
    # convert the hex digest into an integer
    return int(hash_object.hexdigest(), 16)


def extract_year_from_date(text):
    pattern = r'\b((?:19|20)?\d{2})\b'
    current_year = datetime.now().year
    current_year_last_two = int(str(current_year)[-2:])
    match = re.search(pattern, text)

    if match:
        year = match.group(1)
        # Check if the year has two digits
        if len(year) == 2:
            year = int(year)
            # Determine the century
            if year <= current_year_last_two:
                year += 2000
            else:
                year += 1900
            return str(year)
        else:
            return year
    else:
        return "Year not found"


# TODO: align supported formats with Safe Harbor s.t. instead of using generic tag <יום> we will shift the day
def parse_date(date_str):
    # Try to parse full dates
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d.%m.%Y"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            pass
    # Handle year-only and return a datetime object for January 1st of that year
    if re.match(r"^\d{4}$", date_str):
        return datetime(int(date_str), 1, 1)
    # Add more parsing logic here for other formats, e.g., text months
    return None


def get_shift_for_patient(patient_id):
    # consistent random shift based on the patient id
    seed = hash_patient_id(patient_id) % (7 + 7 + 1) - 7  # Ensure the seed is within -7 to +7 range
    return seed


def shift_date(date, shift_days):
    num_date = extract_numerical_date_components(date.text)
    if num_date is None:
        return None
    datetime_obj = datetime(int(num_date.year.text), int(num_date.month.text), int(num_date.day.text))
    new_date = datetime_obj + timedelta(days=shift_days)
    return new_date


def get_date_replacement(date_str, patient_id, mask_operator):
    # medical dates
    if mask_operator == "replace_only_day":
        try:
            date_container = extract_date_components(date_str)
            # in case that the components extraction failed, all values will be none - returning the original text
            if date_container.day is None and date_container.month is None and date_container.year is None:
                return "Date format not supported"

            shift_days = get_shift_for_patient(patient_id)
            shifted_date = shift_date(date_container, shift_days)
            # print (f"patient id: {patient_id}, date: {date_str}, shift: {shift_days}, shifted date: {shifted_date}")
            if shifted_date is None:
                return "Date format is not numerical"
            return shifted_date.strftime("%d.%m.%Y")
        except:
            return "Error while trying to replace date: " + date_str

    # birth dates
    elif mask_operator == "replace_day_month":
        # extract year from the date
        year_extracted = extract_year_from_date(date_str)
        if year_extracted != "Year not found":
            return birth_year_to_year_range(year_extracted)
        else:
            return year_extracted


def format_date(original_format, date):
    if re.match(r"^\d{4}$", original_format):
        return str(date.year)  # Return year-only if original was year-only
    return date.strftime("%d/%m/%Y")


def birth_year_to_year_range(birth_year):
    current_year = datetime.now().year
    age = current_year - int(birth_year)

    if age < 18:
        return f"{current_year - 17} or later"
    elif age <= 25:
        return f"{current_year - 25}-{current_year - 18}"
    elif age > 100:
        return f"Before {current_year - 100}"
    else:
        lower_bound_age = (age - 1) // 5 * 5 + 1  # Rounds down age to nearest boundary, then shifts to start of range
        upper_bound_age = lower_bound_age + 4  # Sets upper bound of age range
        return f"{current_year - upper_bound_age}-{current_year - lower_bound_age}"

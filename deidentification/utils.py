import random
import string

global identifying_prefixes
global exclusion_list

HOSPITAL_LEV_HASHARON = "Lev Hasharon"
HOSPITAL_MAZOR = "Mazor"

HOSPITAL_HEBREW_MAP = {
    HOSPITAL_LEV_HASHARON: "לב השרון",
    HOSPITAL_MAZOR: "מזור"
}

# load identifying prefixes list
def load_identifying_prefixes():
    global identifying_prefixes
    with open("./datasets/identifying_prefix.txt", "r",encoding='utf-8') as file:
        identifying_prefixes = file.read().splitlines()


def load_exclusion_list(filename='./datasets/exclusion_list.txt'):
    global exclusion_list
    with open(filename, 'r', encoding='utf-8') as file:
        exclusion_list = set(line.strip() for line in file)


load_identifying_prefixes()
load_exclusion_list()



def remove_punc(name):
    # remove "-" and "|" at the beginning and end of the name
    name = name.strip("-|")
    # remove ( and )
    name = name.replace("(", "").replace(")", "")
    return name


# def randomize_characters(input_str):
#     def random_digit():
#         return str(random.randint(0, 9))
#
#     result = []
#     for char in input_str:
#         if char.isdigit() or char.isalpha():
#             result.append(random_digit())
#         else:
#             result.append(char)
#     return ''.join(result)

import random


def randomize_characters(input_str):
    def random_digit_exclude(original_digit):
        new_digit = str(random.randint(0, 9))
        # If the generated digit is the same as the original, generate again
        while new_digit == original_digit:
            new_digit = str(random.randint(0, 9))
        return new_digit

    result = []
    for char in input_str:
        if char.isdigit():
            # Replace digits with a random digit that is different from the original
            result.append(random_digit_exclude(char))
        else:
            result.append(char)  # Keep non-digit, non-alpha characters unchanged

    return ''.join(result)


def starts_with_formative_letters(name):
    return name.startswith("ו") or name.startswith("ל") or name.startswith("ב") or name.startswith("כ") \
        or name.startswith("מ") or name.startswith("ה") or name.startswith("ש") or name.startswith("כ")

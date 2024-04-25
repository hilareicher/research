import string


def remove_punc(name):
    name = name.translate(({ord(c): " " for c in string.punctuation}))
    return name


def starts_with_formative_letters(name):
    return name.startswith("ו") or name.startswith("ל") or name.startswith("ב") or name.startswith("כ") \
        or name.startswith("מ") or name.startswith("ה") or name.startswith("ש") or name.startswith("כ")

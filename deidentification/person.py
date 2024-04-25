import pandas as pd
import random
from utils import starts_with_formative_letters, remove_punc

names_df = pd.DataFrame()
replacements = {}


def load_names_data(filename='./datasets/names.csv'):
    global names_df
    # Assuming the CSV file includes headers: Type, Name, Count
    names_df = pd.read_csv(filename).sort_values(by=['Name', 'Count'], ascending=[True, False])
    # Removing duplicates, keeping the entry with the highest count for each name
    names_df = names_df.drop_duplicates(subset='Name', keep='first').reset_index(drop=True)


load_names_data()


def get_name_replacement(name):
    # print (f"get_name_replacement: {name}")
    global names_df, replacements
    name = name.strip()
    if name in ['דר׳', 'דוקטור', 'ד״ר', 'פרופ׳', 'פרופ', 'פרופ׳', 'מדר׳', 'לדר׳', 'דר', "לדר'"]:
        return name

    name = remove_punc(name)
    if name in replacements:
        return replacements[name]

    adjusted_name,formative_letters = adjust_name_if_needed(name)

    if adjusted_name is None:
        print(f"name {name} not found in names dataset")
        return name  # Name not found in the dataset

    if adjusted_name in replacements:
        if formative_letters:
            return formative_letters + replacements[adjusted_name]
        return replacements[adjusted_name]



    # Filter potential replacements to those not already used
    name_type = names_df.loc[names_df['Name'] == adjusted_name, 'Type'].iloc[0]
    potential_replacements = names_df[
        (names_df['Type'] == name_type) &
        (~names_df['Name'].isin(replacements.values())) &
        (names_df['Name'] != adjusted_name)  # Exclude the current name explicitly
        ]['Name'].tolist()

    if not potential_replacements:
        print(f"No unique replacement found for {name}")
        return name
    sampled_name = random.choice(potential_replacements)
    replacements[adjusted_name] = sampled_name
    # if formative_letters were removed, add them back to the sampled name
    if formative_letters:
        return formative_letters + sampled_name

    return sampled_name


def adjust_name_if_needed(name):
    if any(names_df['Name'] == name):
        return name, None

    removed_formatives = ""  # track removed formative letters
    original_name = name  # Keep the original name for reference

    # Attempt to remove formative letters if the name wasn't found
    for _ in range(2):
        if starts_with_formative_letters(name):
            removed_formatives = name[0] + removed_formatives
            name = name[1:]
            if any(names_df['Name'] == name):
                return name, removed_formatives
    return None, None

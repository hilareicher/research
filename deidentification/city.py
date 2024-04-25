import pandas as pd

df = pd.read_csv('./datasets/cities.csv', encoding='utf-8')

selected_cities = {}
# Convert population column to integer to use as weights
df['סה"כ'] = df['סה"כ'].apply(lambda x: int(x.replace(",", "")))


def get_city_replacement(city_name, population_threshold=5000):

    if city_name in ['תל-אביב', 'תל אביב', 'ת"א','תל אביב יפו', 'תל-אביב יפו']:
        selected_cities[city_name] = city_name    # all forms of Tel Aviv, should not be replaced


    # print (f"get_city_replacement: {city_name}")
    if city_name in selected_cities:
        return selected_cities[city_name]
    city_info = df[df['שם_ישוב'] == city_name]

    if city_info.empty:
        random_city_name = df.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
        selected_cities[city_name] = random_city_name
        # print (f"selected random city: {random_city_name}")
        return random_city_name

    # if the city is found but does not meet the population threshold
    elif city_info['סה"כ'].values[0] <= population_threshold:
        # print (f"city {city_name} does not meet the population threshold")
        jewish_or_arab = city_info['מגזר'].values[0]
        location = city_info['פריפריה/ מרכז'].values[0]
        alternative_cities = df[(df['מגזר'] == jewish_or_arab) &
                                (df['פריפריה/ מרכז'] == location)]

        # if there are alternatives, choose one at random
        if not alternative_cities.empty:
            alternative_city_name = alternative_cities.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
            selected_cities[city_name] = alternative_city_name
            #print (f"selected alternative city: {alternative_city_name}")
            return alternative_city_name
        else:
            random_city_name = df.sample(n=1, weights='סה"כ')['שם_ישוב'].values[0]
            selected_cities[city_name] = random_city_name
            #print (f"selected random city: {random_city_name}")
            return random_city_name
    else:
        #print (f"city {city_name} meets the population threshold")
        selected_cities[city_name] = city_name
        return city_name

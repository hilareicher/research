lev_hasharon = 'לב השרון'
lev_hasharon_replacement = 'מוסדנו'


def get_location_replacement(location):
    if lev_hasharon in location:
        return location.replace(lev_hasharon, lev_hasharon_replacement)
    return location

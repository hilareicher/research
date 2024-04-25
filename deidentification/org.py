from utils import starts_with_formative_letters

lev_hasharon = 'לב השרון'
lev_hasharon_replacement = 'מוסדנו'


def get_org_replacement(org_name):
    if lev_hasharon in org_name:
        return org_name.replace(lev_hasharon, lev_hasharon_replacement)
    return "ארגון"
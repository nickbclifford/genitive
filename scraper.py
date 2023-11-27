import requests
from bs4 import BeautifulSoup

BASE_URL = "https://en.wiktionary.org/api/rest_v1/page/html"


def get_declensions(title):
    response = requests.get(f"{BASE_URL}/{title}")
    tree = BeautifulSoup(response.text)

    ru_section = tree.select_one("section:has(#Russian)")

    gen_sg = ru_section.select_one(".gen\|s-form-of").text
    gen_pl = ru_section.select_one(".gen\|p-form-of").text
    
    return (gen_sg, gen_pl)


print(get_declensions("язык"))